import Compiler.main as compiler
from collections import deque

KeyWords: set[str] = {"PROC", "CODE", "STACK", "PROTO", "ENDP", "END", "DWORD", "PTR"}
Operators: set[str] = {"PUSH", "POP", "ADD", "SUB", "IMUL", "DIV", "MOV"}
Registers: set[str] = {"EAX", "EBX", "ECX", "EDX", "ESI", "EBP", "ESP"}


def tokenize(code: str) -> list[list[dict]]:
    code = code.replace('\"', '\'')
    code = code.replace('    ', '\t')

    tokens: list[list[dict]] = [[]]

    inside_name: bool = False
    inside_number: bool = False
    latest: str = ""
    for i in range(len(code)):
        if inside_name:
            if code[i].isalnum() or code[i] == "_":
                latest += code[i]
                continue
            else:
                token_type = "KEYWORD" if latest.upper() in KeyWords else \
                    "OPERATOR" if latest.upper() in Operators else \
                    "REGISTER" if latest.upper() in Registers else "NAME"

                tokens[len(tokens)-1].append({"type": token_type, "value": latest})
                inside_name = False
        elif inside_number:
            if code[i].isnumeric() or code[i] == "." or code[i] == "_":
                latest += code[i]
                continue
            else:
                tokens[len(tokens)-1].append({"type": "NUMBER", "value": latest})
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
                tokens[len(tokens)-1].append({"type": "COLON", "value": ":"})
                continue
            case "+":
                tokens[len(tokens)-1].append({"type": "ADD", "value": "+"})
                continue
            case "-":
                tokens[len(tokens)-1].append({"type": "SUB", "value": "-"})
                continue
            case "[":
                tokens[len(tokens)-1].append({"type": "LSBRACE", "value": "["})
                continue
            case "]":
                tokens[len(tokens)-1].append({"type": "RSBRACE", "value": "]"})
                continue
            case "=":
                tokens[len(tokens)-1].append({"type": "EQUAL", "value": "="})
                continue
            case ";":
                tokens[len(tokens)-1].append({"type": "SEMICOLON", "value": ";"})
                continue
            case ",":
                tokens[len(tokens)-1].append({"type": "COMMA", "value": ","})
                continue
            case ".":
                tokens[len(tokens)-1].append({"type": "DOT", "value": "."})
                continue
            case "\n":
                if len(tokens[len(tokens)-1]) > 0:
                    if tokens[len(tokens) - 1][0]["type"] == "SEMICOLON":
                        tokens[len(tokens) - 1] = []
                    else:
                        tokens.append([])
                continue

    if len(latest) > 0:
        if inside_number:
            tokens[len(tokens)-1].append({"type": "NUMBER", "value": latest})
        elif inside_name:
            token_type = "KEYWORD" if latest.upper() in KeyWords else \
                "OPERATOR" if latest.upper() in Operators else \
                "REGISTER" if latest.upper() in Registers else "NAME"

            tokens[len(tokens) - 1].append({"type": token_type, "value": latest})

    return tokens


class Simulation:
    class RegisterFile:
        eax = 0
        ebx = 0
        ecx = 0
        edx = 0

        ebp = 0
        esi = 0
        esp = 0

        def __getitem__(self, item: str) -> int:
            match item:
                case "eax":
                    return self.eax
                case "ebx":
                    return self.ebx
                case "ecx":
                    return self.ecx
                case "edx":
                    return self.edx
                case "ebp":
                    return self.ebp
                case "esi":
                    return self.esi
                case "esp":
                    return self.esp

        def __setitem__(self, key: str, value: int) -> None:
            match key:
                case "eax":
                    self.eax = value
                case "ebx":
                    self.ebx = value
                case "ecx":
                    self.ecx = value
                case "edx":
                    self.edx = value
                case "ebp":
                    self.ebp = value
                case "esi":
                    self.esi = value
                case "esp":
                    self.esp = value

    current_section: str = "HEADER"
    stack: deque = deque()
    program_counter: int = 0
    function_addresses: dict[str, int] = {}
    entry_point: str = ""
    current_function: str = ""
    registers: RegisterFile = RegisterFile()


def parse_code(tokens: list[list[dict]], token_idx: int) -> None:
    if len(tokens[token_idx]) < 2:
        return

    inst: list[dict] = tokens[token_idx]
    if inst[0]["type"] == "NAME" and inst[1]["type"] == "KEYWORD":
        if inst[1]["value"] == "PROC":
            Simulation.function_addresses[tokens[token_idx][0]["value"]] = token_idx + 1
    elif inst[0]["type"] == "KEYWORD" and inst[1]["type"] == "NAME":
        if inst[0]["value"] == "END":
            Simulation.entry_point = inst[1]["value"]


def get_operand(instruction: list[dict], op_start: int = 0) -> int:
    if instruction[op_start]["type"] == "NUMBER":
        return int(instruction[op_start]["value"])
    elif instruction[op_start]["type"] == "REGISTER":
        return Simulation.registers[instruction[op_start]["value"]]
    elif instruction[op_start]["value"] == "DWORD":
        if instruction[op_start+1]["value"] != "PTR":
            raise Exception("Unexpected Token")

        op_start += 2
        return get_operand(instruction, op_start)
    elif instruction[op_start]["type"] == "LSBRACE":
        val1 = get_operand(instruction, op_start+1)
        if instruction[op_start+2]["value"] == "+":  # Note : Swapped Signs
            val2 = get_operand(instruction, op_start + 3)
            return Simulation.stack[(val1 - val2) // 4 - 1]
        elif instruction[op_start+2]["value"] == "-":  # Note : Swapped Signs
            val2 = get_operand(instruction, op_start + 3)
            return Simulation.stack[(val1 + val2) // 4 - 1]
        else:
            return Simulation.stack[val1 // 4]

    return -1


def interpret_instruction(instruction: list[dict]) -> bool:
    Simulation.program_counter += 1
    match instruction[0]["value"]:
        case "push":
            Simulation.stack.append(get_operand(instruction, 1))
            print(Simulation.stack)
        case "pop":
            Simulation.stack.pop()
            print(Simulation.stack)
        case "mov":
            match instruction[1]["type"]:
                case "REGISTER":
                    Simulation.registers[instruction[1]["value"]] = get_operand(instruction, 3)

        case "ret":
            print(f"RETURN [{Simulation.current_function}] => {Simulation.stack[0]}")
            return False

    return True


def begin_interpret(tokens: list[list[dict]]) -> None:
    Simulation.current_function = Simulation.entry_point
    while interpret_instruction(tokens[Simulation.program_counter]):
        pass


def interpret_program(tokens: list[list[dict]]) -> None:
    # print(*[" ".join(s["value"] for s in q) for q in tokens], sep="\n")

    token_idx: int = 0
    while token_idx < len(tokens):
        if Simulation.current_section == "HEADER":
            print(*[s["value"] for s in tokens[token_idx]], sep=" ")
            if tokens[token_idx][0]["type"] == "DOT":
                if len(tokens[token_idx]) > 1 and tokens[token_idx][1]["value"] == "CODE":
                    Simulation.current_section = "CODE"
        elif Simulation.current_section == "CODE":
            parse_code(tokens, token_idx)

        token_idx += 1

    if Simulation.entry_point not in Simulation.function_addresses.keys():
        raise Exception(f"Entry Point `fn [{Simulation.entry_point}]` not found. ")

    Simulation.instruction = Simulation.function_addresses[Simulation.entry_point]

    print(*[f"fn [{k}] => {Simulation.function_addresses[k]}" for k in Simulation.function_addresses], sep="\n")
    print(f"ENTRY POINT <= fn [{Simulation.entry_point}] => {Simulation.function_addresses[Simulation.entry_point]}")

    begin_interpret(tokens)


def interpret(input_file_path: str) -> None:
    with open(input_file_path, mode="r") as output_file_path:
        interpret_program(tokenize(output_file_path.read()))


def main():
    input_file_path: str = "C:\\dev\\Code\\Python\\$PycharmProjects\\Compiler\\test\\main.code"
    output_file_path: str = "C:\\dev\\Code\\Python\\$PycharmProjects\\Compiler\\test\\main.asm"
    # compiler.run(input_file_path, output_file_path)
    interpret(output_file_path)


if __name__ == "__main__":
    main()
