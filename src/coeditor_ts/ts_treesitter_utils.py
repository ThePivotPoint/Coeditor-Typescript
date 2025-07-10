from tree_sitter import Parser
from tree_sitter_languages import get_language
from typing import List, Dict, Any
from dataclasses import dataclass

TS_LANGUAGE = get_language('typescript')

@dataclass
class NodeInfo:
    type: str
    name: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    # 可扩展更多字段

@dataclass
class Change:
    change_type: str  # 'added', 'deleted', 'modified'
    node: NodeInfo
    # 可扩展更多字段

def parse_ts_file_with_treesitter(path: str):
    """解析 TypeScript 文件为 AST"""
    parser = Parser()
    parser.set_language(TS_LANGUAGE)
    with open(path, 'r', encoding='utf-8') as f:
        code = f.read()
    tree = parser.parse(bytes(code, 'utf8'))
    return tree, code

def extract_functions_classes(tree, code: str) -> List[NodeInfo]:
    """遍历 AST，提取所有 function/class/method 节点及其范围"""
    node_infos = []
    root = tree.root_node
    def visit(node):
        # 只提取 function/class/method
        if node.type in ('function_declaration', 'class_declaration', 'method_definition'):
            # 获取名称
            name = ''
            for child in node.children:
                if child.type == 'identifier':
                    name = code[child.start_byte:child.end_byte]
                    break
            node_infos.append(NodeInfo(
                type=node.type,
                name=name,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
            ))
        for child in node.children:
            visit(child)
    visit(root)
    return node_infos

def diff_ast(tree_old, code_old: str, tree_new, code_new: str) -> List[Change]:
    """对比两份 AST，找出结构变更（如函数/类的增删改）"""
    nodes_old = extract_functions_classes(tree_old, code_old)
    nodes_new = extract_functions_classes(tree_new, code_new)
    # 简单 diff: 以 (type, name) 为 key
    old_keys = {(n.type, n.name): n for n in nodes_old}
    new_keys = {(n.type, n.name): n for n in nodes_new}
    changes = []
    for k, n in new_keys.items():
        if k not in old_keys:
            changes.append(Change('added', n))
        else:
            # 可扩展: 检查内容是否有变化
            pass
    for k, n in old_keys.items():
        if k not in new_keys:
            changes.append(Change('deleted', n))
    return changes 