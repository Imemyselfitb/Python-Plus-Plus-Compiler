# TOKEN TYPES: "NAME", "NUMBER", "COLON", "ADD", "SUB", "MULT", "DIV", "INDENT", "DEDENT"

KeyWords: set[str] = {"return", "def", "exit", }
Types: set[str] = {"i32", }
Operators: set[str] = {"ADD", "SUB", "MULT", "DIV", }


def tokenize(code: str) -> list[dict]:
    code = code.replace('\"', '\'')
    code = code.replace('    ', '\t')

    tokens: list[dict] = []

    inside_name: bool = False
    inside_number: bool = False
    latest: str = ""
    indent_level: int = 0
    for i in range(len(code)):
        # Handle Indenting
        if code[i-1] == "\n":
            # There is one MORE Indent
            while code[i:i + indent_level + 1] == "\t" * (indent_level + 1):
                tokens.append({"type": "INDENT", "value": indent_level})
                indent_level += 1
                continue  # Char was an indent (can continue to next character)

            # There is one LESS Indent
            while code[i:i + indent_level] != "\t" * indent_level:
                tokens.append({"type": "DEDENT", "value": indent_level})
                indent_level -= 1

        if inside_name:
            if code[i].isalnum() or code[i] == "_":
                latest += code[i]
                continue
            else:
                tokens.append({"type": "KEYWORD" if latest in KeyWords else "NAME", "value": latest})
                inside_name = False
        elif inside_number:
            if code[i].isnumeric() or code[i] == "." or code[i] == "_":
                latest += code[i]
                continue
            else:
                tokens.append({"type": "NUMBER", "value": latest})
                inside_number = False
        elif code[i].isnumeric():
            inside_number = True
            latest = code[i]
            continue
        elif code[i].isalpha():
            inside_name = True
            latest = code[i]
            continue

        match (code[i]):
            case ":":
                tokens.append({"type": "COLON", "value": ":"})
                continue
            case "+":
                tokens.append({"type": "ADD", "value": "+"})
                continue
            case "-":
                tokens.append({"type": "SUB", "value": "-"})
                continue
            case "*":
                tokens.append({"type": "MULT", "value": "*"})
                continue
            case "/":
                tokens.append({"type": "DIV", "value": "/"})
                continue
            case "(":
                tokens.append({"type": "LPAREN", "value": "("})
                continue
            case ")":
                tokens.append({"type": "RPAREN", "value": ")"})
                continue
            case "=":
                tokens.append({"type": "EQUAL", "value": "="})
                continue
            case ";":
                tokens.append({"type": "SEMICOLON", "value": ";"})
                continue
            case ",":
                tokens.append({"type": "COMMA", "value": ","})
                continue
            case ".":
                tokens.append({"type": "DOT", "value": "."})
                continue

    if len(latest) > 0:
        if inside_number:
            tokens.append({"type": "NUMBER", "value": latest})
        elif inside_name:
            tokens.append({"type": "NAME", "value": latest})

    return tokens


class Node:
    def __init__(self, source_code_parent_node, data: dict):
        self.parent = source_code_parent_node
        self.children: list = []
        self.data: dict = data
        self.source_code: str = ""

    def add_child(self, child: object) -> None:
        self.children.append(child)

    def add_source_code(self, source: str):
        if len(self.source_code) < 1:
            self.source_code = source
        else:
            self.source_code += " " + source

        if self.parent:
            self.parent.add_source_code(source)

    def __str__(self, offset: int = 1) -> str:
        output: str = "\t" * (offset - 1) + "{\n"
        for key in self.data:
            if type(self.data[key]) is str:
                output += ("\t" * offset) + key + ": \"" + self.data[key] + "\",\n"
            elif self.data[key] is not None:
                output += ("\t" * offset) + key + ": " + str(self.data[key]) + ",\n"

        if len(self.children) > 0:
            output += ("\t" * offset) + "children: [\n"
            for child in self.children:
                if child == self:
                    output += ("\t" * (offset + 1)) + "@Parent,\n"
                    continue
                output += child.__str__(offset + 2) + ",\n"
            output += ("\t" * offset) + "]\n"

        output += "\t" * (offset - 1) + "}"
        return output


def parse_params(tokens: list[dict], parent_node: Node, i: int) -> int:
    while tokens[i]["type"] != "RPAREN" and i < len(tokens):
        if i < len(tokens) - 1 and tokens[i+1]["type"] in Operators:
            i = parse_expr(tokens, parent_node, i, skip_parenthesis=False)
        else:
            i = parse_value(tokens, parent_node, i)

        if tokens[i]["type"] == "COMMA":
            parent_node.add_source_code(tokens[i]["value"])
            i += 1

    parent_node.add_source_code(tokens[i]["value"])

    return i


def parse_value(tokens: list[dict], parent_node: Node, i: int) -> int:
    if tokens[i]["type"] == "NUMBER":
        node: Node
        if "." in tokens[i]["value"]:
            node = Node(parent_node, {"type": "Number", "value": float(tokens[i]["value"])})
        else:
            node = Node(parent_node, {"type": "Number", "value": int(tokens[i]["value"])})

        node.add_source_code(tokens[i]["value"])
        parent_node.add_child(node)
        return i + 1

    if tokens[i]["type"] == "NAME":
        node: Node
        if i < len(tokens) - 1 and tokens[i + 1]["type"] == "LPAREN":
            node = Node(parent_node, {"type": "FunctionCall", "value": tokens[i]["value"]})
            node.add_source_code(tokens[i]["value"])
            i += 1
            node.add_source_code(tokens[i]["value"])
            i += 1
            i = parse_params(tokens, node, i)
            parent_node.add_child(node)
        else:
            node = Node(parent_node, {"type": "Variable", "value": tokens[i]["value"]})
            node.add_source_code(tokens[i]["value"])
            parent_node.add_child(node)

        return i + 1

    if tokens[i]["type"] == "LPAREN":
        return parse_expr(tokens, parent_node, i + 1)

    raise ValueError("Tried to get Value of `" + tokens[i]["value"] + "`: " + str(tokens[i]))


def parse_term(tokens: list[dict], parent_node: Node, i: int) -> int:
    if i >= len(tokens):
        return i

    node: Node = Node(parent_node, {"type": "BinaryExpression", "value": None})

    if tokens[i]["type"] == "LPAREN":
        node.add_source_code(tokens[i]["value"])
        i = parse_expr(tokens, node, i + 1)
    else:
        i = parse_value(tokens, node, i)

    if i >= len(tokens) or (tokens[i]["type"] != "MULT" and tokens[i]["type"] != "DIV"):
        for c in node.children:
            parent_node.add_child(c)
        return i

    node.add_source_code(tokens[i]["value"])
    node.data["value"] = tokens[i]["value"]
    i += 1

    if tokens[i]["type"] == "LPAREN":
        i = parse_expr(tokens, node, i + 1)
    else:
        i = parse_value(tokens, node, i)

    while i < len(tokens):
        iterative_parent: Node = Node(parent_node, {"type": "BinaryExpression", "value": None})

        if tokens[i]["type"] != "MULT" and tokens[i]["type"] != "DIV":
            break

        iterative_parent.data["value"] = tokens[i]["value"]
        i += 1

        iterative_parent.add_child(node)
        if tokens[i]["type"] == "LPAREN":
            i = parse_expr(tokens, iterative_parent, i + 1) + 1
        else:
            i = parse_value(tokens, iterative_parent, i)

        node = iterative_parent

    parent_node.add_child(node)
    return i


def parse_expr(tokens: list[dict], parent_node: Node, i: int, skip_parenthesis: bool = True) -> int:
    node: Node = Node(parent_node, {"type": "BinaryExpression", "value": None})
    i = parse_term(tokens, node, i)

    if i >= len(tokens):
        for c in node.children:
            parent_node.add_child(c)
        return i

    if tokens[i]["type"] != "ADD" and tokens[i]["type"] != "SUB":
        for c in node.children:
            parent_node.add_child(c)

        if skip_parenthesis and tokens[i]["type"] == "RPAREN":
            parent_node.add_source_code(tokens[i]["value"])
            i += 1

        return i

    node.add_source_code(tokens[i]["value"])
    node.data["value"] = tokens[i]["value"]
    i = parse_term(tokens, node, i + 1)

    while i < len(tokens):
        iterative_parent: Node = Node(node, {"type": "BinaryExpression", "value": None})
        if tokens[i]["type"] != "ADD" and tokens[i]["type"] != "SUB":
            if tokens[i]["type"] == "RPAREN":
                iterative_parent.add_source_code(tokens[i]["value"])
                i += 1
            break

        iterative_parent.add_source_code(tokens[i]["value"])
        iterative_parent.data["value"] = tokens[i]["value"]
        i += 1

        iterative_parent.add_child(node)
        i = parse_term(tokens, iterative_parent, i)
        node = iterative_parent

    parent_node.add_child(node)
    return i


def parse_return(tokens: list[dict], parent_node: Node, i: int, exiting: bool) -> int:
    node: Node = Node(parent_node, {"type": "Exit" if exiting else "Return", "value": None})
    node.add_source_code(tokens[i]["value"])
    i += 1

    try:
        i = parse_expr(tokens, node, i)
    finally:
        parent_node.add_child(node)
        return i


def parse_var_assign(tokens: list[dict], parent_node: Node, i: int) -> int:
    node: Node = Node(parent_node, {"type": "Assignment", "value": None})
    i = parse_value(tokens, node, i)

    if tokens[i]["type"] != "EQUAL":
        raise Exception("Expected '=' After Variable Name")

    node.add_source_code(tokens[i]["value"])

    i = parse_expr(tokens, node, i + 1)

    parent_node.add_child(node)
    return i


def parse_body(tokens: list[dict], parent_node: Node, i: int) -> int:
    while i < len(tokens) and tokens[i]["type"] != "DEDENT":
        prev_token_idx: int = i
        i = parse_statement(tokens, parent_node, i)

        if prev_token_idx == i:
            i += 1

    return i


def parse_function_params(tokens: list[dict], parent_node: Node, i: int) -> int:
    param: Node = Node(parent_node, {"type": None, "value": None})

    if tokens[i]["type"] == "RPAREN":
        param.add_source_code(tokens[i]["value"])
        return i + 1

    if tokens[i]["type"] != "NAME":
        raise Exception("Expected Parameter Name")

    param.add_source_code(tokens[i]["value"])
    param.data["value"] = tokens[i]["value"]
    i += 1

    if tokens[i]["type"] != "COLON":
        raise Exception("Expected ':' after Parameter Name")

    param.add_source_code(tokens[i]["value"])
    i += 1

    if tokens[i]["type"] != "NAME":
        raise Exception("Expected ':' after Parameter Name")

    param.add_source_code(tokens[i]["value"])
    param.data["type"] = tokens[i]["value"]
    i += 1

    parent_node.add_child(param)

    if tokens[i]["type"] == "COMMA":
        param.add_source_code(tokens[i]["value"])
        return parse_function_params(tokens, parent_node, i + 1)

    if tokens[i]["type"] != "RPAREN":
        raise Exception("Expected ')'")

    param.add_source_code(tokens[i]["value"])

    return i + 1


def parse_function(tokens: list[dict], parent_node: Node, i: int) -> int:
    node: Node = Node(parent_node, {"type": "Function", "value": None})
    node.add_source_code(tokens[i]["value"])
    i += 1

    if tokens[i]["type"] != "NAME":
        raise Exception("Expected Function Name after DEF")

    node.add_source_code(tokens[i]["value"])
    node.data["value"] = tokens[i]["value"]
    i += 1

    if tokens[i]["type"] != "LPAREN":
        raise Exception("Expected '(' After Function Name")

    node.add_source_code(tokens[i]["value"])
    i += 1

    params_node: Node = Node(node, {"type": "FunctionParams", "value": None})
    i = parse_function_params(tokens, params_node, i)
    node.add_child(params_node)

    if tokens[i]["type"] != "COLON":
        raise Exception("Expected ':' At the end of Function Declaration")

    i += 1

    if tokens[i]["type"] != "INDENT":
        raise Exception("Expected INDENT")

    i += 1

    body_node: Node = Node(None, {"type": "FunctionBody", "value": None})
    i = parse_body(tokens, body_node, i)
    node.add_child(body_node)

    parent_node.add_child(node)

    return i


def parse_statement(tokens: list[dict], parent_node: Node, i: int) -> int:
    if tokens[i]["type"] == "KEYWORD" and tokens[i]["value"] == "exit":
        i = parse_return(tokens, parent_node, i, exiting=True)
    elif tokens[i]["type"] == "KEYWORD" and tokens[i]["value"] == "return":
        i = parse_return(tokens, parent_node, i, exiting=False)
    elif tokens[i]["type"] == "NAME" and i < len(tokens) - 1 and tokens[i + 1]["type"] == "EQUAL":
        i = parse_var_assign(tokens, parent_node, i)
    elif tokens[i]["type"] == "KEYWORD" and tokens[i]["value"] == "def":
        i = parse_function(tokens, parent_node, i)
    elif tokens[i]["type"] == "NAME" and i < len(tokens) - 1 and tokens[i + 1]["type"] == "LPAREN":
        i = parse_expr(tokens, parent_node, i)

    return i


def parse(tokens: list[dict]) -> Node:
    abstract_tree: Node = Node(None, {"type": "Root", "value": None})

    token_idx: int = 0
    while token_idx < len(tokens):
        prev_token_idx: int = token_idx
        token_idx = parse_statement(tokens, abstract_tree, token_idx)

        if prev_token_idx == token_idx:
            token_idx += 1

    return abstract_tree


GlobalVariables: dict[str:int] = {}
GlobalsStackSize: int = 0
CompiledFunctionExtension: str = "_CompiledFunction"

FunctionStackSize: dict[str, int] = {}
FunctionVariables: dict[str, dict[str: int]] = {}


def check_same_contents(node1: Node, node2: Node):
    if node1.data != node2.data or len(node1.children) != len(node2.children):
        return False

    for i in range(len(node1.children)):
        if not check_same_contents(node1.children[i], node2.children[i]):
            return False

    return True


def compile_function_call_params(function_call_node: Node, function_name: str | None = None) -> str:
    output = ""
    function_call_node.children.reverse()
    for c in function_call_node.children:
        var_value: str = "eax"
        if c.data["type"] == "BinaryExpression":
            output += compile_binary_expression(c, function_name)
        elif c.data["type"] == "FunctionCall":
            output += compile_binary_expression(c, function_name)
        elif c.data["type"] == "Variable":
            var_value = compile_binary_expression(c, function_name)
        else:
            var_value = compile_binary_expression(c, function_name)

        output += f"\tpush {var_value}\n"

    return output


def compile_function_call(func_call_node: Node, function_name: str | None = None) -> str:
    output: str = compile_function_call_params(func_call_node, function_name)
    if not function_name:
        output += "\tmov esi, ebp\n"
    output += f"\tcall {func_call_node.data['value']}{CompiledFunctionExtension}\n"
    output += f"\tadd esp, {len(func_call_node.children) * 4}\n"
    return output


def compile_binary_expression(bin_exp: Node, function_name: str | None = None) -> str:
    if bin_exp.data["type"] == "Number":
        return str(bin_exp.data["value"])
    elif bin_exp.data["type"] == "Variable":
        if bin_exp.data['value'] in GlobalVariables:
            if function_name:
                return f"DWORD PTR [esi{GlobalVariables[bin_exp.data['value']]}]"
            else:
                return f"DWORD PTR [ebp{GlobalVariables[bin_exp.data['value']]}]"
        elif function_name:
            return f"DWORD PTR [ebp{FunctionVariables[function_name][bin_exp.data['value']]}]"
    elif bin_exp.data["type"] == "FunctionCall":
        return compile_function_call(bin_exp, function_name)

    output: str = ""
    second_op: str = "edx"

    if bin_exp.children[1].data["type"] == "BinaryExpression" or bin_exp.children[1].data["type"] == "FunctionCall":
        output += compile_binary_expression(bin_exp.children[1], function_name)
        output += "\tmov ecx, eax\n"
        second_op = "ecx"

    if bin_exp.children[0].data["type"] == "BinaryExpression":
        output += compile_binary_expression(bin_exp.children[0], function_name)
        if bin_exp.children[1].data["type"] == "BinaryExpression":
            output += "\tmov eax, ecx\n"
    elif bin_exp.children[0].data["type"] == "FunctionCall":
        output += compile_binary_expression(bin_exp.children[0], function_name)
    elif bin_exp.children[1].data["type"] == "BinaryExpression":
        output += "\tmov ecx, " + compile_binary_expression(bin_exp.children[0], function_name) + "\n"
    else:
        output += "\tmov eax, " + compile_binary_expression(bin_exp.children[0], function_name) + "\n"

    if check_same_contents(bin_exp.children[0], bin_exp.children[1]):
        second_op = "eax"
    elif bin_exp.children[1].data["type"] == "Number":
        second_op = bin_exp.children[1].data["value"]
    elif bin_exp.children[1].data["type"] == "Variable":
        output += "\tmov edx, " + compile_binary_expression(bin_exp.children[1], function_name) + "\n"

    match(bin_exp.data["value"]):
        case "+":
            output += f"\tadd eax, {second_op}\n"
        case "-":
            output += f"\tsub eax, {second_op}\n"
        case "*":
            output += f"\timul eax, {second_op}\n"
        case "/":
            output += f"\tdiv eax, {second_op}\n"

    return output


def compile_var_assignment(assign_node: Node, function_name: str | None = None):
    global GlobalVariables, GlobalsStackSize

    output: str = f"\n\t; [[ {assign_node.source_code} ]]\n"
    var_name: str = assign_node.children[0].data["value"]
    if function_name:
        if var_name not in FunctionVariables[function_name] and var_name not in GlobalVariables:
            var_size: int = 4  # Will eventually be read in file
            FunctionStackSize[function_name] += var_size
            FunctionVariables[function_name][var_name] = f"-{FunctionStackSize[function_name]}"
    elif var_name not in GlobalVariables:
        var_size: int = 4  # Will eventually be read in file
        GlobalsStackSize += var_size
        GlobalVariables[var_name] = f"-{GlobalsStackSize}"

    var_value: str = "eax"
    if assign_node.children[1].data["type"] == "BinaryExpression":
        output += compile_binary_expression(assign_node.children[1], function_name)
    elif assign_node.children[1].data["type"] == "FunctionCall":
        output += compile_binary_expression(assign_node.children[1], function_name)
    elif assign_node.children[1].data["type"] == "Variable":
        output += f"\tmov ebx, {compile_binary_expression(assign_node.children[1], function_name)}\n"
        var_value = "ebx"
    else:
        var_value = compile_binary_expression(assign_node.children[1], function_name)

    if var_name not in GlobalVariables:
        output += f"\tmov DWORD PTR [ebp{FunctionVariables[function_name][var_name]}], {var_value}\n"
    else:
        output += f"\tmov DWORD PTR [ebp{GlobalVariables[var_name]}], {var_value}\n"
    return output


def compile_return(return_node: Node, function_name: str) -> str:
    output: str = f"\n\t; [[ {return_node.source_code} ]]\n"
    node_type: str = return_node.children[0].data["type"]
    if node_type == "BinaryExpression" or node_type == "FunctionCall":
        output += compile_binary_expression(return_node.children[0], function_name)
        output += "\n\t; Clean up Stack \n\tmov esp, ebp\n\tpop ebp\n\tret\n"
        return output
    elif node_type == "Variable":
        var_value: str
        if return_node.children[0].data['value'] in GlobalVariables:
            var_value = f"DWORD PTR [esi{GlobalVariables[return_node.children[0].data['value']]}]"
        else:
            var_value: str = f"DWORD PTR [ebp{FunctionVariables[function_name][return_node.children[0].data['value']]}]"

        return f"{output}\tmov eax, {var_value}\n\n\t; Clean up Stack \n\tmov esp, ebp\n\tpop ebp\n\tret\n"
    else:
        var_value: str = compile_binary_expression(return_node.children[0], function_name)
        return f"{output}\tmov eax, {var_value}\n\n\t; Clean up Stack \n\tmov esp, ebp\n\tret\n"


def compile_exit(exit_node: Node, function_name: str | None = None) -> str:
    output: str = f"\n\t; [[ {exit_node.source_code} ]]\n"
    if len(exit_node.children) < 1:
        return f"{output}\tpop ebp\n\tpush 0\n\tcall ExitProcess\n\tret\n"

    if exit_node.children[0].data["type"] == "BinaryExpression" or exit_node.children[0].data["type"] == "FunctionCall":
        output += compile_binary_expression(exit_node.children[0], function_name)
        output += "\tpop ebp\n\tpush eax\n\tcall ExitProcess\n\tret\n"
        return output
    else:
        var_value: str = compile_binary_expression(exit_node.children[0], function_name)
        return f"{output}\tpop ebp\n\tpush {var_value}\n\tcall ExitProcess\n\tret\n"


def get_type_size(type_name: str) -> int:
    match type_name:
        case "i32":
            return 4

    return 0


def compile_function_params(function_params_node: Node, function_name: str) -> None:
    params_stack_size: int = 4
    FunctionStackSize[function_name] = 0
    FunctionVariables[function_name] = {}
    for param in function_params_node.children:
        params_stack_size += get_type_size(param.data["type"])
        FunctionVariables[function_name][param.data["value"]] = f"+{params_stack_size}"


def compile_body(
        function_body_node: Node,
        function_name: str | None = None,
        input_string: str | None = None
) -> tuple[str, bool, bool]:
    program: str = ""
    if input_string:
        program += input_string

    has_returned: bool = False
    has_exited: bool = False
    for c in function_body_node.children:
        match c.data["type"]:
            case "Assignment":
                program = program + compile_var_assignment(c, function_name)
            case "Exit":
                program = program + compile_exit(c, function_name)
                has_exited = True
            case "Return":
                program = program + compile_return(c, function_name)
                has_returned = True
            case "Function":
                program = compile_function(c) + program
            case "FunctionCall":
                program = program + f"\n\t; [[ {c.source_code} ]]\n"
                program = program + compile_function_call(c, function_name)

    return program, has_returned, has_exited


def compile_function(function_node: Node) -> str:
    output: str = f"; [[ {function_node.source_code} ]]\n"
    output += f"{function_node.data['value']}{CompiledFunctionExtension} PROC\n"
    output += "\tpush ebp\n\tmov ebp, esp\n"

    param_child_index: int = 0
    body_child_index: int = 0
    for i, node in enumerate(function_node.children):
        if node.data["type"] == "FunctionParams":
            param_child_index = i
        elif node.data["type"] == "FunctionBody":
            body_child_index = i

    compile_function_params(function_node.children[param_child_index], function_node.data["value"])
    body_output, has_returned, _ = compile_body(function_node.children[body_child_index], function_node.data["value"])

    if not has_returned:
        body_output += "\n\t; Clean up Stack \n\tmov esp, ebp\n\tpop ebp\n\tret\n"

    output += body_output

    output += f"{function_node.data['value']}{CompiledFunctionExtension} ENDP\n\n"
    return output


def try_solve_expressions(expression: Node) -> None:
    if expression.data["type"] != "BinaryExpression":
        for c in expression.children:
            try_solve_expressions(c)
        return

    if len(expression.children) < 2:
        return

    if expression.children[0].data["type"] != "Number":
        try_solve_expressions(expression.children[0])
        if expression.children[0].data["type"] != "Number":
            try_solve_expressions(expression.children[1])
            return

    if expression.children[1].data["type"] != "Number":
        try_solve_expressions(expression.children[1])
        if expression.children[1].data["type"] != "Number":
            return

    expression.data["type"] = "Number"

    arg1 = expression.children[0].data["value"]
    arg2 = expression.children[1].data["value"]

    match(expression.data["value"]):
        case "+":
            expression.data["value"] = arg1 + arg2
        case "-":
            expression.data["value"] = arg1 - arg2
        case "*":
            expression.data["value"] = arg1 * arg2
        case "/":
            if type(arg1) is int and type(arg2) is int:
                expression.data["value"] = arg1 // arg2
            else:
                expression.data["value"] = arg1 / arg2

    expression.children.clear()


def compile_program(abstract_tree: Node) -> str:
    try_solve_expressions(abstract_tree)

    output = ".386\n.model flat, stdcall\n.stack 4096\n\n"
    output += "ExitProcess PROTO, dwExitCode:DWORD\n\n"
    output += ".CODE\n\n"

    program, _, has_exited = compile_body(abstract_tree, None, "main PROC\n\tpush ebp\n")

    if not has_exited:
        program += "\tpop ebp\n\tpush 0\n\tcall ExitProcess\n\tret\n"

    output += program + "main ENDP\n\nEND main\n"

    return output


def run(input_file_path: str, output_file_path: str) -> None:
    with open(input_file_path, mode="r") as input_file:
        with open(output_file_path, mode="w") as output_file:
            parsed_result = parse(tokenize(input_file.read()))
            output_file.write(compile_program(parsed_result))

    print("Code Created!")


def run_verbose(input_file_path: str, output_file_path: str) -> None:
    with open(input_file_path, mode="r") as input_file:
        with open(output_file_path, mode="w") as output_file:
            parsed_result = parse(tokenize(input_file.read()))
            print(parsed_result)
            print(*[(">>>> " + c.source_code) for c in parsed_result.children], sep="\n")
            output_file.write(compile_program(parsed_result))

    print("Code Created!")


if __name__ == "__main__":
    InputFilePath: str = "C:\\dev\\Code\\Python\\$PycharmProjects\\Compiler\\test\\main.code"
    OutputFilePath: str = "C:\\dev\\Code\\MASM\\MASM\\Main.asm"
    run_verbose(InputFilePath, OutputFilePath)
