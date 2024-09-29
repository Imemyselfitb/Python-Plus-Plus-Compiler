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
                tokens.append({"type": "SEMICOLON", "value": "="})
                continue
            case ",":
                tokens.append({"type": "COMMA", "value": ","})
                continue

    if len(latest) > 0:
        if inside_number:
            tokens.append({"type": "NUMBER", "value": latest})
        elif inside_name:
            tokens.append({"type": "NAME", "value": latest})

    # print(*tokens, sep="\n")

    return tokens


class Node:
    def __init__(self, data: dict):
        self.children = []
        self.data = data

    def add_child(self, child: object) -> None:
        self.children.append(child)

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
            i = parse_expr(tokens, parent_node, i, skipParenthesis=False)
        else:
            i = parse_value(tokens, parent_node, i)

        if tokens[i]["type"] == "COMMA":
            i += 1

    return i


def parse_value(tokens: list[dict], parent_node: Node, i: int) -> int:
    if tokens[i]["type"] == "NUMBER":
        node: Node
        if "." in tokens[i]["value"]:
            node = Node({"type": "Number", "value": float(tokens[i]["value"])})
        else:
            node = Node({"type": "Number", "value": int(tokens[i]["value"])})
        parent_node.add_child(node)
        return i + 1

    if tokens[i]["type"] == "NAME":
        node: Node
        if i < len(tokens) - 1 and tokens[i + 1]["type"] == "LPAREN":
            node = Node({"type": "FunctionCall", "value": tokens[i]["value"]})
            i += 2
            i = parse_params(tokens, node, i)
            parent_node.add_child(node)
        else:
            node = Node({"type": "Variable", "value": tokens[i]["value"]})
            parent_node.add_child(node)
        return i + 1

    if tokens[i]["type"] == "LPAREN":
        return parse_expr(tokens, parent_node, i + 1)

    raise ValueError("Tried to get Value of `" + tokens[i]["value"] + "`: " + str(tokens[i]))


def parse_term(tokens: list[dict], parent_node: Node, i: int) -> int:
    if i >= len(tokens):
        return i

    node: Node = Node({"type": "BinaryExpression", "value": None})

    if tokens[i]["type"] == "LPAREN":
        i = parse_expr(tokens, node, i + 1)
    else:
        i = parse_value(tokens, node, i)

    if i >= len(tokens) or (tokens[i]["type"] != "MULT" and tokens[i]["type"] != "DIV"):
        for c in node.children:
            parent_node.add_child(c)
        return i

    node.data["value"] = tokens[i]["value"]
    i += 1

    if tokens[i]["type"] == "LPAREN":
        i = parse_expr(tokens, node, i + 1)
    else:
        i = parse_value(tokens, node, i)

    while i < len(tokens):
        iterative_parent: Node = Node({"type": "BinaryExpression", "value": None})

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


def parse_expr(tokens: list[dict], parent_node: Node, i: int, skipParenthesis: bool = True) -> int:
    node: Node = Node({"type": "BinaryExpression", "value": None})
    i = parse_term(tokens, node, i)

    if i >= len(tokens):
        for c in node.children:
            parent_node.add_child(c)
        return i

    if tokens[i]["type"] != "ADD" and tokens[i]["type"] != "SUB":
        for c in node.children:
            parent_node.add_child(c)

        if skipParenthesis and tokens[i]["type"] == "RPAREN":
            i += 1

        return i

    node.data["value"] = tokens[i]["value"]
    i = parse_term(tokens, node, i + 1)

    while i < len(tokens):
        iterative_parent: Node = Node({"type": "BinaryExpression", "value": None})
        if tokens[i]["type"] != "ADD" and tokens[i]["type"] != "SUB":
            if tokens[i]["type"] == "RPAREN":
                i += 1
            break
        iterative_parent.data["value"] = tokens[i]["value"]
        i += 1

        iterative_parent.add_child(node)
        i = parse_term(tokens, iterative_parent, i)
        node = iterative_parent

    parent_node.add_child(node)
    return i


def parse_return(tokens: list[dict], parent_node: Node, i: int, exiting: bool) -> int:
    node: Node = Node({"type": "Exit" if exiting else "Return", "value": None})
    i += 1
    try:
        i = parse_expr(tokens, node, i)
    finally:
        parent_node.add_child(node)
        return i


def parse_var_assign(tokens: list[dict], parent_node: Node, i: int) -> int:
    node: Node = Node({"type": "Assignment", "value": None})
    i = parse_value(tokens, node, i)

    if tokens[i]["type"] != "EQUAL":
        raise Exception("Expected '=' After Variable Name")

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
    param: Node = Node({"type": None, "value": None})

    if tokens[i]["type"] != "NAME":
        raise Exception("Expected Parameter Name")

    param.data["value"] = tokens[i]["value"]
    i += 1

    if tokens[i]["type"] != "COLON":
        raise Exception("Expected ':' after Parameter Name")
    i += 1

    if tokens[i]["type"] != "NAME":
        raise Exception("Expected ':' after Parameter Name")

    param.data["type"] = tokens[i]["value"]
    i += 1

    parent_node.add_child(param)

    if tokens[i]["type"] == "COMMA":
        return parse_function_params(tokens, parent_node, i + 1)

    if tokens[i]["type"] != "RPAREN":
        raise Exception("Expected ')'")

    return i + 1


def parse_function(tokens: list[dict], parent_node: Node, i: int) -> int:
    i += 1

    if tokens[i]["type"] != "NAME":
        raise Exception("Expected Function Name after DEF")

    node: Node = Node({"type": "Function", "value": tokens[i]["value"]})
    i += 1

    if tokens[i]["type"] != "LPAREN":
        raise Exception("Expected '(' After Function Name")
    i += 1

    params_node: Node = Node({"type": "FunctionParams", "value": None})
    i = parse_function_params(tokens, params_node, i)
    node.add_child(params_node)

    if tokens[i]["type"] != "COLON":
        raise Exception("Expected ':' At the end of Function Declaration")
    i += 1

    if tokens[i]["type"] != "INDENT":
        raise Exception("Expected INDENT")
    i += 1

    body_node: Node = Node({"type": "FunctionBody", "value": None})
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

    return i


def parse(tokens: list[dict]) -> Node:
    abstract_tree: Node = Node({"type": "Root", "value": None})

    token_idx: int = 0
    while token_idx < len(tokens):
        prev_token_idx: int = token_idx
        token_idx = parse_statement(tokens, abstract_tree, token_idx)

        if prev_token_idx == token_idx:
            token_idx += 1

    return abstract_tree


Variables: dict[str:int] = {}
CurrentStackSize: int = 0
CompiledFunctionExtension: str = "_CompiledFunction"


def check_same_contents(node1: Node, node2: Node):
    if node1.data != node2.data or len(node1.children) != len(node2.children):
        return False

    for i in range(len(node1.children)):
        if not check_same_contents(node1.children[i], node2.children[i]):
            return False

    return True


def compile_function_call_params(function_call_node: Node) -> str:
    output = ""
    function_call_node.children.reverse()
    for c in function_call_node.children:
        var_value: str = "eax"
        if c.data["type"] == "BinaryExpression":
            output += compile_binary_expression(c)
        elif c.data["type"] == "FunctionCall":
            output += compile_binary_expression(c)
        elif c.data["type"] == "Variable":
            var_value = compile_binary_expression(c)
        else:
            var_value = compile_binary_expression(c)

        output += f"push {var_value}\n"

    return output


def compile_binary_expression(bin_exp: Node) -> str:
    if bin_exp.data["type"] == "Number":
        return str(bin_exp.data["value"])
    elif bin_exp.data["type"] == "Variable":
        return f"DWORD PTR [{Variables[bin_exp.data['value']]}]"
    elif bin_exp.data["type"] == "FunctionCall":
        return f"{compile_function_call_params(bin_exp)}call {bin_exp.data['value']}{CompiledFunctionExtension}\n"

    output: str = ""
    second_op: str = "edx"

    if bin_exp.children[1].data["type"] == "BinaryExpression":
        output += compile_binary_expression(bin_exp.children[1])
        output += "mov ecx, eax\n"
        second_op = "ecx"
    elif bin_exp.children[1].data["type"] == "FunctionCall":
        output += compile_binary_expression(bin_exp.children[1])
        output += "mov ecx, eax\n"
        second_op = "ecx"

    if bin_exp.children[0].data["type"] == "BinaryExpression":
        output += compile_binary_expression(bin_exp.children[0])
        # if bin_exp.children[1].data["type"] == "BinaryExpression":
        #     output += "mov eax, ecx\n"
    elif bin_exp.children[0].data["type"] == "FunctionCall":
        output += compile_binary_expression(bin_exp.children[0])
    elif bin_exp.children[1].data["type"] == "BinaryExpression":
        output += "mov ecx, " + compile_binary_expression(bin_exp.children[0]) + "\n"
    else:
        output += "mov eax, " + compile_binary_expression(bin_exp.children[0]) + "\n"

    if check_same_contents(bin_exp.children[0], bin_exp.children[1]):
        second_op = "eax"
    elif bin_exp.children[1].data["type"] == "Number":
        second_op = bin_exp.children[1].data["value"]
    elif bin_exp.children[1].data["type"] == "Variable":
        output += "mov edx, " + compile_binary_expression(bin_exp.children[1]) + "\n"

    match(bin_exp.data["value"]):
        case "+":
            output += f"add eax, {second_op}\n"
        case "-":
            output += f"sub eax, {second_op}\n"
        case "*":
            output += f"imul eax, {second_op}\n"
        case "/":
            output += f"div eax, {second_op}\n"

    return output


def compile_var_assignment(assign_node: Node):
    global Variables, CurrentStackSize

    output: str = ""
    var_name: str = assign_node.children[0].data["value"]
    if var_name not in Variables:
        var_size: int = 4  # Will eventually be read in file
        CurrentStackSize += var_size
        Variables[var_name] = f"ebp-{CurrentStackSize}"

    var_value: str = "eax"
    if assign_node.children[1].data["type"] == "BinaryExpression":
        output += compile_binary_expression(assign_node.children[1])
    elif assign_node.children[1].data["type"] == "BinaryExpression":
        output += compile_binary_expression(assign_node.children[1])
    elif assign_node.children[1].data["type"] == "Variable":
        output += f"mov ebx, {compile_binary_expression(assign_node.children[1])}\n"
        var_value = "ebx"
    else:
        var_value = compile_binary_expression(assign_node.children[1])

    output += f"mov DWORD PTR [{Variables[var_name]}], {var_value}\n"
    return output


def compile_return(return_node: Node) -> str:
    if return_node.children[0].data["type"] == "BinaryExpression":
        output: str = compile_binary_expression(return_node.children[0])
        output += "mov esp, ebp\npop ebp\nret\n"
        return output
    elif return_node.children[0].data["type"] == "FunctionCall":
        output: str = compile_binary_expression(return_node.children[0])
        output += "mov esp, ebp\npop ebp\nret\n"
        return output
    elif return_node.children[0].data["type"] == "Variable":
        var_value: str = f"DWORD PTR [{Variables[return_node.children[0].data['value']]}]"
        return f"mov eax, {var_value}\nmov esp, ebp\npop ebp\nret\n"
    else:
        var_value: str = compile_binary_expression(return_node.children[0])
        return f"mov eax, {var_value}\nmov esp, ebp\npop ebp\nret\n"


def compile_exit(exit_node: Node) -> str:
    if len(exit_node.children) < 1:
        return f"pop ebp\npush 0\ncall ExitProcess\n"

    if exit_node.children[0].data["type"] == "BinaryExpression":
        output: str = compile_binary_expression(exit_node.children[0])
        output += "pop ebp\npush eax\ncall ExitProcess\n"
        return output
    elif exit_node.children[0].data["type"] == "FunctionCall":
        output: str = compile_binary_expression(exit_node.children[0])
        output += "pop ebp\npush eax\ncall ExitProcess\n"
        return output
    else:
        var_value: str = compile_binary_expression(exit_node.children[0])
        return f"pop ebp\npush {var_value}\ncall ExitProcess\n"


def get_type_size(type_name: str) -> int:
    match type_name:
        case "i32":
            return 4

    return 0


FunctionStackSize: dict[str, int] = {}


def compile_function_params(function_params_node: Node, function_name: str) -> None:
    global FunctionStackSize
    FunctionStackSize[function_name] = 4
    for param in function_params_node.children:
        FunctionStackSize[function_name] += get_type_size(param.data["type"])
        Variables[param.data["value"]] = f"ebp+{FunctionStackSize[function_name]}"


def compile_body(function_body_node: Node) -> str:
    program: str = ""
    for c in function_body_node.children:
        match c.data["type"]:
            case "Assignment":
                program = program + compile_var_assignment(c)
            case "Exit":
                program = program + compile_exit(c)
            case "Return":
                program = program + compile_return(c)
            case "Function":
                program = compile_function(c) + program
    return program


def compile_function(function_node: Node) -> str:
    output: str = f"{function_node.data['value']}{CompiledFunctionExtension} PROC\npush ebp\nmov ebp, esp\n"

    param_child_index: int = 0
    body_child_index: int = 0
    for i, node in enumerate(function_node.children):
        if node.data["type"] == "FunctionParams":
            param_child_index = i
        elif node.data["type"] == "FunctionBody":
            body_child_index = i

    compile_function_params(function_node.children[param_child_index], function_node.data["value"])
    output += compile_body(function_node.children[body_child_index])

    output += f"{function_node.data['value']}{CompiledFunctionExtension} ENDP\n"
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


def compile_program(abstract_tree: Node, verbose: bool = True) -> str:
    try_solve_expressions(abstract_tree)

    output = ".386\n.model flat, stdcall\n.stack 4096\n"
    output += "ExitProcess PROTO, dwExitCode:DWORD\n"
    output += ".CODE\n"

    program = "main PROC\npush ebp\n"

    exited_value: bool = False

    for c in abstract_tree.children:
        if verbose:
            output += ""
        match c.data["type"]:
            case "Assignment":
                program = program + compile_var_assignment(c)
            case "Exit":
                program = program + compile_exit(c)
                exited_value = True
            case "Function":
                program = compile_function(c) + program

    if not exited_value:
        program += "pop ebp\npush 0\ncall ExitProcess\n"

    output += program + "main ENDP\nEND main\n"

    return output


def main() -> None:
    input_file_path: str = "C:\\dev\\Code\\Python\\$PycharmProjects\\Compiler\\test\\main.code"
    output_file_path: str = "C:\\dev\\Code\\MASM\\MASM\\Main.asm"

    with open(input_file_path, mode="r") as input_file:
        with open(output_file_path, mode="w") as output_file:
            parsed_result = parse(tokenize(input_file.read()))
            print(parsed_result)
            output_file.write(compile_program(parsed_result))

    print("Code Created!")


if __name__ == "__main__":
    main()
    # input("Exiting...")
