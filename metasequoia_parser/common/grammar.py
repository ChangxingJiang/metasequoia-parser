"""
语法解析器的语法管理器

为了处理 移进/规约冲突，我们采用如下方案：
1. 在定义 Grammar 时，允许标记终结符的 SR 优先级（即 Shift/Reduce 优先级）并规定结合方向，同时允许在每个规则中标记 SR 优先级（使用终结符 SR
优先级定义，并继承终结符的结合方向）。如果没有规定结合方向，则结合方向默认为从左到右。
2. 在构造 Grammar 时，会根据终结符 SR 优先级和规则 SR 优先级计算每个产生式的 SR 优先级。具体地，我们会优先尝试使用规则的 SR 优先级；如果规则
没有定义 SR 优先级，则会遍历（不递归）规则中的所有终结符，并将其中 SR 优先级最高的终结符的 SR 优先级以及结合方向作为规则的 SR 优先级和结合方向；
如果规则中没有终结符，或规则中的所有终结符都没有定义 SR 优先级，则会将规则的 SR 优先级置为默认值（最小值），同时将结合方向置为默认值（即从左到右）。
3. 在构造 ACTION + GOTO 表时，如果出现 移进/规约冲突：
  - 如果要规约的规则的 SR 优先级高于下一个输入符号的 SR 优先级，则进行规约；
  - 如果要规约的规则的 SR 优先级低于下一个输入符号的 SR 优先级，则进行移进；
  - 如果要规约的规则的 SR 优先级与下一个输入符号的 SR 优先级一致，即均使用同一个终结符的 SR 优先级，则根据该符号的结合方向：
    - 如果结合方向为从左到右，则进行规约
    - 如果结合方向为从右到左，则进行移进

为了处理 规约/规约冲突，我们采用如下方案：
1. 在定义 Grammar 时，允许标记规则的 RR 优先级（即 Reduce/Reduce 优先级）
2. 在构造 Grammar 时，会根据规则的 RR 优先级和规则的出现顺序计算每个产生式的 RR 优先级。具体地，优先使用规则的 RR 优先级，当规则的 RR 优先级
相同时，使用规则的先后顺序。
3. 在构造 ACTION + GOTO 表时，如果出现 规约/规约冲突，则优先规约其中 RR 优先级更高的产生式。
"""

import collections
import dataclasses
import enum
import inspect
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Type, Union

from metasequoia_parser.common.symbol import Symbol, TerminalType
from metasequoia_parser.exceptions import GrammarError

__all__ = [
    "Grammar",
    "GrammarBuilder",
    "Product",
    "SRPriority",
    "CombineType",
    "GGroup",
    "GRule",
    "GrammarActionParams"
]


class GrammarActionParams:
    """语法定义器：备选规则行为参数"""

    def __init__(self, symbols: List[Symbol]):
        """构造方法

        Parameters
        ----------
        symbols : List[Symbol]
            所有参数，按从前往后的顺序
        """
        self._symbols = symbols

    @property
    def n_param(self) -> int:
        """返回参数数量"""
        return len(self._symbols)

    def symbols(self) -> List[Symbol]:
        """获取所有参数的列表"""
        return self._symbols

    def symbol(self, i: int) -> Symbol:
        """获取第 i 个参数的 Symbol 对象"""
        return self._symbols[i]

    def value(self, i: int) -> Any:
        """获取第 i 个参数的值"""
        return self._symbols[i].symbol_value

    __getitem__ = value


DEFAULT_ACTION: Callable[[GrammarActionParams], Any] = lambda x: x[0]


@dataclasses.dataclass(slots=True, frozen=True, eq=True)
class GRule:
    """语法定义器：语义组的备选规则（原始）"""

    symbols: Tuple[str, ...] = dataclasses.field(kw_only=True, hash=True, compare=True)  # 备选规则中的符号名称列表
    action: Callable[[GrammarActionParams], Any] = dataclasses.field(kw_only=True)  # 备选规则的规约行为
    sr_priority_as: Optional[str] = dataclasses.field(kw_only=True, default=None)  # 备选规则的 SR 优先级（视作与哪个终结符一致）
    rr_priority: int = dataclasses.field(kw_only=True, default=0)  # 备选规则的 RR 优先级

    @staticmethod
    def create(symbols: List[Union[str, TerminalType]],
               action: Callable[[GrammarActionParams], Any] = DEFAULT_ACTION,
               sr_priority_as: Optional[Union[str, TerminalType]] = None,
               rr_priority: int = 0) -> "GRule":
        """语义组的备选规则 Rule 对象的构造方法"""
        # 检查 symbols 是否为 List[str] 的类型
        if not isinstance(symbols, list):
            raise GrammarError("备选规则的 symbols 参数不是列表类型")
        format_symbols = []
        for symbol in symbols:
            if isinstance(symbol, str):
                format_symbols.append(symbol)
            elif isinstance(symbol, TerminalType):
                format_symbols.append(symbol.name)
            else:
                assert isinstance(symbol, str), "备选规则的 symbols 列表中的元素不是字符串类型或 TerminalType 类型"

        # 检查 action 是否可调用，是否有且仅有一个参数
        assert callable(action), "备选规则的 action 参数不支持调用"
        inspect_action = inspect.signature(action)
        assert len(inspect_action.parameters) == 1, f"备选规则的 action 函数参数数量为 {len(inspect_action.parameters)}"

        # 检查 sr_priority_as 参数
        if isinstance(sr_priority_as, TerminalType):
            sr_priority_as = sr_priority_as.name

        # 检查 rr_priority 优先级
        assert isinstance(rr_priority, int), "RR 优先级的类型不是整型"
        assert rr_priority >= 0, "RR 优先级为小于 0 的数"

        # 构造 Rule 对象
        return GRule(
            symbols=tuple(format_symbols),
            action=action,
            sr_priority_as=sr_priority_as,
            rr_priority=rr_priority
        )


@dataclasses.dataclass(slots=True)
class GGroup:
    """语法解析器的语义组"""

    name: str = dataclasses.field(kw_only=True)
    rules: List[GRule] = dataclasses.field(kw_only=True)

    @staticmethod
    def create(name: str, rules: List[GRule]):
        """语义组 Group 对象的构造方法"""
        # 检查 name 是字符串类型且不为空
        assert isinstance(name, str), "语义组的 name 参数不是字符串类型"
        assert name != "", "语义组的 name 参数为空"

        # 检查 rules 是 List[Rule] 类型
        assert isinstance(rules, list), "备选规则的 rules 参数不是列表类型"
        for rule in rules:
            assert isinstance(rule, GRule), "备选规则的 rules 列表中的元素不是 Rule 类型"

        # 构造 Group 对象
        return GGroup(
            name=name,
            rules=rules
        )


class CombineType(enum.Enum):
    """结合方法"""

    LEFT = 1  # 指定一个或多个符号为左结合
    RIGHT = 2  # 指定一个或多个符号为右结合
    NONASSOC = 3  # 指定一个或多个符号既不是左结合也不是右结合，通常用于比较运算符等，以防止出现诸如 `a < b < c` 的表达式产生歧义


@dataclasses.dataclass(slots=True, frozen=True, eq=True)
class SRPriority:
    """SR 优先级（移进/规约优先级）"""

    terminal_name_list: List[str] = dataclasses.field(kw_only=True)  # 终结符名称
    combine_type: CombineType = dataclasses.field(kw_only=True, default=CombineType.LEFT)  # 终结符结合方法

    @staticmethod
    def create(symbols: List[Union[str, TerminalType]], combine_type: CombineType):
        """SR 优先级的构造方法"""

        # 检查 symbols 是否为 List[str] 的类型
        assert isinstance(symbols, list), "SR 优先级规则的 symbols 参数不是列表类型"
        format_symbols = []
        for symbol in symbols:
            if isinstance(symbol, str):
                format_symbols.append(symbol)
            elif isinstance(symbol, TerminalType):
                format_symbols.append(symbol.name)
            else:
                assert isinstance(symbol, str), "SR 优先级规则的 symbols 列表中的元素不是字符串类型或 TerminalType 类型"
        return SRPriority(
            terminal_name_list=format_symbols,
            combine_type=combine_type
        )


@dataclasses.dataclass(slots=True, frozen=True)
class Product:
    """语法类生成式"""

    nonterminal_id: int = dataclasses.field(kw_only=True)  # 语义组的名称
    symbol_id_list: List[int] = dataclasses.field(kw_only=True)  # 生成式中的元素
    action: Callable = dataclasses.field(kw_only=True)  # 生成式的执行逻辑
    sr_priority_idx: int = dataclasses.field(kw_only=True)  # 生成式的 SR 优先级序号（越大越优先）
    sr_combine_type: CombineType = dataclasses.field(kw_only=True)  # 生成式的 SR 合并顺序
    rr_priority_idx: int = dataclasses.field(kw_only=True)  # 生成式的 RR 优先级序号（越大越优先）


@dataclasses.dataclass(slots=True)
class GrammarBuilder:
    """语法类构造器"""

    start: str = dataclasses.field(kw_only=True)  # 入口语义组名称
    groups: List[GGroup] = dataclasses.field(kw_only=True)  # 语义组列表
    terminal_type_enum: Type[TerminalType] = dataclasses.field(kw_only=True)  # 终结符类型的枚举类
    sr_priority: List[SRPriority] = dataclasses.field(kw_only=True, default_factory=lambda: [])  # SR 优先级列表（越靠前越优先）

    @staticmethod
    def create(start: str,
               terminal_type_enum: Type[TerminalType],
               groups: Optional[List[GGroup]] = None,
               sr_priority: Optional[List[SRPriority]] = None) -> "GrammarBuilder":
        """语法类构造器的构造方法"""
        if groups is None:
            groups = []
        if sr_priority is None:
            sr_priority = []
        return GrammarBuilder(
            start=start,
            groups=groups,
            terminal_type_enum=terminal_type_enum,
            sr_priority=sr_priority
        )

    def group_append(self, group: GGroup) -> "GrammarBuilder":
        """添加语义组"""
        self.groups.append(group)
        return self

    def build(self) -> "Grammar":
        """语法类 Grammar 的构造方法"""
        # 获取终结符和非终结符的名称集合
        nonterminal_name_set = {group.name for group in self.groups}
        terminal_name_set = {item.name for item in self.terminal_type_enum}

        # 检查是否存在重名的标识符、非标识符
        assert len(nonterminal_name_set) == len(self.groups), "存在重名的多个语义组"
        assert len(terminal_name_set & nonterminal_name_set) == 0, "存在语义组与终结符名称相同"

        # 检查入口语义组是否已被定义
        assert self.start in nonterminal_name_set, "入口语义组未被定义"

        # 检查每个语义组中使用的符号是否均被定义
        symbol_name_set = nonterminal_name_set | terminal_name_set
        for group in self.groups:
            for rule_idx, rule in enumerate(group.rules):
                for symbol_name in rule.symbols:
                    if symbol_name not in symbol_name_set:
                        raise GrammarError(f"语义组 {group.name} 的第 {rule_idx} 个备选规则中"
                                           f"的 {symbol_name} 符号名称未被定义")

        # 构造 Grammar 对象
        return Grammar(
            entry_group_name=self.start,
            group_list=self.groups,
            terminal_type_enum=self.terminal_type_enum,
            sr_priority_list=self.sr_priority,
        )


class Grammar:
    # pylint: disable=R0902
    """语法类"""

    @dataclasses.dataclass(slots=True, frozen=True, eq=True)
    class _SRPriority:
        """SR 优先级（移进/规约优先级）"""

        priority_idx: int = dataclasses.field(kw_only=True)  # SR 优先级顺序（越大越优先）
        combine_type: CombineType = dataclasses.field(kw_only=True, default=CombineType.LEFT)  # 终结符结合方法

    # 【常量】入口虚拟符号名称
    WRAPPER_ENTRANCE_SYMBOL_NAME = "S'"

    # ------------------------------ 语法类构造方法 ------------------------------

    def __init__(self,
                 entry_group_name: str,
                 group_list: List[GGroup],
                 terminal_type_enum: Type[TerminalType],
                 sr_priority_list: List[SRPriority]):

        self._terminal_type_enum = terminal_type_enum

        # 【初始化】终结符名称到 ID 的映射表（作为 ACTION 表的列下标）
        self._terminal_name_id_hash = self._create_terminal_name_id_hash(terminal_type_enum)
        self._n_terminal = len(self._terminal_name_id_hash)

        # 【初始化】非终结符名称到 ID 的映射表（作为 GOTO 表的列下标）
        self._nonterminal_name_id_hash = self._create_nonterminal_name_id_hash(group_list, self._n_terminal)
        self._n_nonterminal = len(self._nonterminal_name_id_hash)

        # 【初始化】符号名称（包括非终结符和终结符）到 ID 的映射表（作为 ACTION 表 + GOTO 表的列下标）
        self._symbol_name_id_hash = self._create_symbol_name_id_hash(self._terminal_name_id_hash,
                                                                     self._nonterminal_name_id_hash)
        self._symbol_id_name_hash = self._create_symbol_id_name_hash(self._symbol_name_id_hash)
        self._n_symbol = len(self._symbol_name_id_hash)

        # 【计算】终结符 ID 名称到终结符 SR 优先级的映射
        self._terminal_id_sr_priority_hash: Dict[int, Grammar._SRPriority] = self._cal_terminal_id_sr_priority_hash(
            sr_priority_list=sr_priority_list,
            terminal_name_id_hash=self._terminal_name_id_hash,
        )

        # 【计算】备选规则（规约非终结符名称、 备选规则 Rule 对象的元组）到 RR 优先级的映射
        self._rule_rr_priority_hash = self._cal_nonterminal_id_rr_priority_hash(
            group_list=group_list,
            entrance_symbol_name=self.WRAPPER_ENTRANCE_SYMBOL_NAME,
            entry_group_name=entry_group_name
        )
        # print("self._nonterminal_id_rr_priority_hash:", self._rule_rr_priority_hash)

        # 【初始化】生成式列表
        self._product_list: List[Product] = []

        # 增广语义，增加入口产生式
        self._product_list.append(self.create_product(
            nonterminal_name=self.WRAPPER_ENTRANCE_SYMBOL_NAME,
            symbol_name_tuple=(entry_group_name,),
            action=lambda x: x,
            sr_priority_as=None
        ))

        # 从语义组中解析出所有产生式
        for group in group_list:
            for rule in group.rules:
                self._product_list.append(self.create_product(
                    nonterminal_name=group.name,
                    symbol_name_tuple=rule.symbols,
                    action=rule.action,
                    sr_priority_as=rule.sr_priority_as
                ))

        # 构造相关映射关系
        self._name_product_hash = {}
        for product in self._product_list:
            # print(product)
            if product.nonterminal_id not in self._name_product_hash:
                self._name_product_hash[product.nonterminal_id] = []
            self._name_product_hash[product.nonterminal_id].append(product)

        # 【计算】可能为空的非终结符（id）集合
        self._maybe_empty_nonterminal_set: Set[int] = set()
        for product in self._product_list:
            if len(product.symbol_id_list) == 0:
                self._maybe_empty_nonterminal_set.add(product.nonterminal_id)

    def create_product(self,
                       nonterminal_name: str,
                       symbol_name_tuple: Tuple[str, ...],
                       action: Callable,
                       sr_priority_as: Optional[str]) -> Product:
        """构造产生式"""
        # 根据非终结符的名称获取非终结符的 ID
        nonterminal_id = self._nonterminal_name_id_hash[nonterminal_name]

        # 根据每个符号的名称，获取备选规则中每个符号的 ID
        symbol_id_list = []
        for symbol in symbol_name_tuple:
            if symbol in self._terminal_name_id_hash:
                symbol_id_list.append(self._terminal_name_id_hash[symbol])
            else:
                symbol_id_list.append(self._nonterminal_name_id_hash[symbol])

        # 计算生成式的 SR 优先级
        if sr_priority_as is not None:
            # 优先尝试使用规则的 SR 优先级
            sr_terminal_id = self._terminal_name_id_hash[sr_priority_as]
            sr_priority_idx = self.get_terminal_sr_priority_idx(sr_terminal_id)
            sr_combine_type = self.get_terminal_sr_combine_type(sr_terminal_id)
        else:
            # 如果规则没有定义 SR 优先级，则会遍历（不递归）规则中的所有终结符，并将其中 SR 优先级最高的终结符的 SR 优先级以及结合方向作为规则的 SR 优先级和结合方向
            # 如果规则中没有终结符，或规则中的所有终结符都没有定义 SR 优先级，则会将规则的 SR 优先级置为默认值（最小值），同时将结合方向置为默认值（即从左到右）
            sr_priority_idx = 0
            sr_combine_type = CombineType.LEFT
            for symbol_id in symbol_id_list:
                if not self.is_terminal(symbol_id):
                    continue  # 跳过非终结符
                if self.get_terminal_sr_priority_idx(symbol_id) > sr_priority_idx:
                    sr_priority_idx = self.get_terminal_sr_priority_idx(symbol_id)
                    sr_combine_type = self.get_terminal_sr_combine_type(symbol_id)

        # 计算生成式的 RR 优先级
        rr_priority_idx = self._rule_rr_priority_hash[(nonterminal_name, symbol_name_tuple)]

        return Product(
            nonterminal_id=nonterminal_id,
            symbol_id_list=symbol_id_list,
            action=action,
            sr_priority_idx=sr_priority_idx,
            sr_combine_type=sr_combine_type,
            rr_priority_idx=rr_priority_idx
        )

    # ------------------------------ 计算、获取非终结符名称和终结符名称到 ID 的映射表和数量 ------------------------------

    @staticmethod
    def _create_terminal_name_id_hash(terminal_type_enum: Type[TerminalType]) -> Dict[str, int]:
        """计算终结符名称到 ID 的映射表（作为 ACTION 表的列下标）"""
        return {item.name: item.value for item in terminal_type_enum}

    @staticmethod
    def _create_nonterminal_name_id_hash(groups: List[GGroup], n_terminal: int) -> Dict[str, int]:
        """计算非终结符名称到 ID 的映射表（作为 GOTO 表的列下标）"""
        nonterminal_name_id_hash = {}
        for i, group in enumerate(groups):
            nonterminal_name = group.name
            if nonterminal_name in nonterminal_name_id_hash:
                raise GrammarError(f"存在重名的非终结符 {nonterminal_name}")
            nonterminal_name_id_hash[nonterminal_name] = n_terminal + i

        # 将外层的虚拟入口语义组添加到非终结符中
        if Grammar.WRAPPER_ENTRANCE_SYMBOL_NAME in nonterminal_name_id_hash:
            raise GrammarError(f"存在与虚拟入口语义组重名的非终结符 {Grammar.WRAPPER_ENTRANCE_SYMBOL_NAME}")
        nonterminal_name_id_hash[Grammar.WRAPPER_ENTRANCE_SYMBOL_NAME] = n_terminal + len(nonterminal_name_id_hash)

        return nonterminal_name_id_hash

    @staticmethod
    def _create_symbol_name_id_hash(terminal_name_id_hash: Dict[str, int],
                                    nonterminal_name_id_hash: Dict[str, int]) -> Dict[str, int]:
        """计算符号名称（包含非终结符和终结符）到符号 ID 的映射表"""
        symbol_name_id_hash = {}
        for terminal_name, terminal_id in terminal_name_id_hash.items():
            symbol_name_id_hash[terminal_name] = terminal_id
        for nonterminal_name, nonterminal_id in nonterminal_name_id_hash.items():
            assert nonterminal_name not in symbol_name_id_hash, f"存在重名的终结符和非终结符 {nonterminal_name}"
            symbol_name_id_hash[nonterminal_name] = nonterminal_id
        return symbol_name_id_hash

    @staticmethod
    def _create_symbol_id_name_hash(symbol_name_id_hash: Dict[str, int]) -> List[str]:
        """计算符号 ID 到符号名称的映射表"""
        symbol_id_name_hash: List[Optional[str]] = [None] * len(symbol_name_id_hash)
        for symbol_name, symbol_id in symbol_name_id_hash.items():
            symbol_id_name_hash[symbol_id] = symbol_name
        return symbol_id_name_hash

    @property
    def n_nonterminal(self) -> int:
        """获取非终结符的数量"""
        return self._n_nonterminal

    @property
    def n_terminal(self) -> int:
        """获取终结符的数量"""
        return self._n_terminal

    @property
    def symbol_name_id_hash(self) -> Dict[str, int]:
        """获取符号名称（包括非终结符和终结符）到 ID 的映射表（作为 ACTION 表 + GOTO 表的列下标）"""
        return self._symbol_name_id_hash

    @property
    def n_symbol(self) -> int:
        """获取符号数量"""
        return self._n_symbol

    @property
    def terminal_type_enum(self) -> Type[TerminalType]:
        """返回终结符枚举类"""
        return self._terminal_type_enum

    # ------------------------------ SR 优先级和 RR 优先级计算函数 ------------------------------

    @staticmethod
    def _cal_terminal_id_sr_priority_hash(sr_priority_list: List[SRPriority],
                                          terminal_name_id_hash: Dict[str, int]
                                          ) -> Dict[int, "Grammar._SRPriority"]:
        """计算终结符 ID 名称到终结符 SR 优先级的映射"""
        terminal_id_sr_priority_hash: Dict[int, Grammar._SRPriority] = {}

        # 添加主动定义的终结符优先级映射
        for i, sr_priority in enumerate(sr_priority_list):
            for terminal_name in sr_priority.terminal_name_list:
                terminal_id = terminal_name_id_hash[terminal_name]
                terminal_id_sr_priority_hash[terminal_id] = Grammar._SRPriority(
                    priority_idx=len(sr_priority_list) - i,
                    combine_type=sr_priority.combine_type
                )

        # 添加其他终结符的优先级映射
        for terminal_name, terminal_id in terminal_name_id_hash.items():
            if terminal_id not in terminal_id_sr_priority_hash:
                terminal_id_sr_priority_hash[terminal_id] = Grammar._SRPriority(
                    priority_idx=0,
                    combine_type=CombineType.LEFT  # 默认向左合并
                )

        return terminal_id_sr_priority_hash

    @staticmethod
    def _cal_nonterminal_id_rr_priority_hash(group_list: List[GGroup],
                                             entrance_symbol_name: str,
                                             entry_group_name: str
                                             ) -> Dict[Tuple[str, Tuple[str, ...]], int]:
        """计算备选规则（规约非终结符名称、备选规则 Rule 对象的元组）到 RR 优先级的映射"""
        # 聚合每个优先级的备选规则
        rr_priority_hash = collections.defaultdict(list)
        for group in group_list:
            for rule in group.rules:
                rr_priority_hash[rule.rr_priority].append((group.name, rule))

        # 根据优先级和先后顺序，构造所有备选规则按优先级排序降序排序的列表
        rr_priority_list = []
        for rr_priority in sorted(rr_priority_hash.keys(), reverse=True):
            rr_priority_list.extend(rr_priority_hash[rr_priority])
        nonterminal_id_rr_priority_hash = {(group_name, rule.symbols): len(rr_priority_list) - i
                                           for i, (group_name, rule) in enumerate(rr_priority_list)}

        # 添加入口语义组优先级
        nonterminal_id_rr_priority_hash[(entrance_symbol_name, (entry_group_name,))] = 0

        return nonterminal_id_rr_priority_hash

    # ------------------------------ 符号属性查询方法 ------------------------------

    @property
    def entrance_symbol(self) -> int:
        """获取入口虚拟符号"""
        return self._nonterminal_name_id_hash[self.WRAPPER_ENTRANCE_SYMBOL_NAME]

    def is_entrance_symbol(self, symbol: int):
        """如果 symbol 是入口虚拟符号则返回 True，否则返回 False"""
        return symbol == self.entrance_symbol

    @property
    def end_terminal(self) -> int:
        """获取结束符对应的终结符"""
        return 0

    def get_symbol_name(self, symbol_id: int) -> str:
        """根据符号 ID 获取符号名称"""
        return self._symbol_id_name_hash[symbol_id]

    def get_terminal_sr_priority_idx(self, terminal_id: int) -> int:
        """获取 terminal_id 的 SR 优先级序号（越大越优先）"""
        return self._terminal_id_sr_priority_hash[terminal_id].priority_idx

    def get_terminal_sr_combine_type(self, terminal_id: int) -> CombineType:
        """获取 terminal_id 的 SR 合并方向"""
        return self._terminal_id_sr_priority_hash[terminal_id].combine_type

    # ------------------------------ 产生式列表 ------------------------------

    def get_product_list(self) -> List[Product]:
        """返回生成式的列表"""
        return self._product_list

    # ------------------------------ 符号特性查询函数 ------------------------------

    def is_terminal(self, symbol: int) -> bool:
        """如果 id 是终结符则返回 True，否则返回 False"""
        return symbol < self.n_terminal

    def is_maybe_empty(self, symbol: int) -> bool:
        """如果 symbol 为非终结符，且可能匹配 %empty 则返回 True；否则返回 False"""
        return symbol in self._maybe_empty_nonterminal_set


if __name__ == "__main__":
    print(TerminalType.create_by_terminal_name_list("TestTerminalType", ["a", "b", "c"]))
