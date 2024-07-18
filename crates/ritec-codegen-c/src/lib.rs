use std::collections::HashMap;

use ritec_mir::{
    Block, BlockId, Body, BodyId, Const, ConstKind, Local, Operand, Place, ProjectionKind,
    Statement, Terminator, Type, Unit, Value,
};

struct Codegen<'a> {
    unit: &'a Unit,
    define: String,
    bodies: String,
    structs: HashMap<Vec<Type>, String>,
    unions: HashMap<Vec<Type>, String>,
    functions: HashMap<Vec<Type>, String>,
}

impl Codegen<'_> {
    fn gen_type(&mut self, ty: &Type) -> String {
        match ty {
            Type::Bool => String::from("uint8_t"),
            Type::Int { signed, width } => match width {
                Some(width) => {
                    if *signed {
                        format!("int{}_t", width)
                    } else {
                        format!("uint{}_t", width)
                    }
                }
                None => if *signed { "ptrdiff_t" } else { "size_t" }.to_string(),
            },
            Type::Float { width } => match width {
                32 => "float".to_string(),
                64 => "double".to_string(),
                _ => unimplemented!(),
            },
            Type::Pointer { mutable, pointee } => {
                let pointee = self.gen_type(pointee);

                if *mutable {
                    format!("{}*", pointee)
                } else {
                    format!("const {}*", pointee)
                }
            }
            Type::Array { element, length } => {
                let element = self.gen_type(element);

                format!("{}[{}]", element, length)
            }
            Type::Function { arguments, output } => {
                let mut types = arguments.clone();
                types.push(output.as_ref().clone());

                if let Some(name) = self.functions.get(&types) {
                    return name.clone();
                }

                let output = self.gen_type(output);

                let mut args = String::new();

                for (i, argument) in arguments.iter().enumerate() {
                    if i > 0 {
                        args += ", ";
                    }

                    args += &self.gen_type(argument);
                }

                let name = format!("_rite_function_{}", self.functions.len());

                self.define += &format!("typedef {} (*{})({});\n", output, name, args);
                self.define += "\n";

                self.functions.insert(types, name.clone());

                name
            }
            Type::Struct { fields } => {
                if let Some(name) = self.structs.get(fields) {
                    return name.clone();
                }

                let types: Vec<_> = fields.iter().map(|ty| self.gen_type(ty)).collect();

                self.define += &format!("// struct {:?}\n", types);
                self.define += "typedef struct {\n";

                for (i, ty) in types.iter().enumerate() {
                    self.define += &format!("    {} {};\n", ty, field_name(i));
                }

                let name = format!("_rite_struct_{}", self.structs.len());

                self.define += &format!("}} {};\n", name);
                self.define += "\n";

                self.structs.insert(fields.clone(), name.clone());

                name
            }
            Type::Union { variants } => {
                if let Some(name) = self.unions.get(variants) {
                    return name.clone();
                }

                let types: Vec<_> = variants.iter().map(|ty| self.gen_type(ty)).collect();

                self.define += &format!("// union {:?}\n", types);
                self.define += "typedef union {\n";

                for (i, ty) in types.iter().enumerate() {
                    self.define += &format!("    {} {};\n", ty, variant_name(i));
                }

                let name = format!("_rite_union_{}", self.unions.len());

                self.define += &format!("}} {};\n", name);
                self.define += "\n";

                self.unions.insert(variants.clone(), name.clone());

                name
            }
        }
    }

    fn gen_place(&mut self, place: &Place) -> String {
        let mut code = local_name(place.base);

        for projection in &place.projections {
            match projection.kind {
                ProjectionKind::Deref => code = format!("*{}", code),
                ProjectionKind::Field(index) => code = format!("({}).{}", code, field_name(index)),
            }
        }

        code
    }

    fn gen_const(&mut self, constant: &Const) -> String {
        match constant.kind {
            ConstKind::Int(v) => {
                if matches!(constant.ty, Type::Pointer { .. }) {
                    return format!("(void*) {}", v);
                }

                format!("{}", v)
            }
            ConstKind::Float(v) => format!("{}", v),
            ConstKind::Struct(ref fields) => {
                let ty = self.gen_type(&constant.ty);

                let fields: Vec<_> = fields
                    .iter()
                    .map(|field| {
                        // dont one-line please
                        self.gen_const(field)
                    })
                    .collect();

                format!("({}) {{ {} }}", ty, fields.join(", "))
            }
            ConstKind::Body(body_id) => format!("({})", body_name(body_id)),
        }
    }

    fn gen_operand(&mut self, operand: &Operand) -> String {
        match operand {
            Operand::Copy(place) => self.gen_place(place),
            Operand::Move(place) => self.gen_place(place),
            Operand::Const(constant) => self.gen_const(constant),
        }
    }

    fn gen_value(&mut self, value: &Value) -> String {
        match value {
            Value::Use(operand) => self.gen_operand(operand),
            Value::Binary(op, lhs, rhs) => {
                let op = match op {
                    ritec_mir::BinaryOp::Add => "+",
                    ritec_mir::BinaryOp::Sub => "-",
                    ritec_mir::BinaryOp::Mul => "*",
                    ritec_mir::BinaryOp::Div => "/",
                    ritec_mir::BinaryOp::Rem => "%",
                    ritec_mir::BinaryOp::Eq => "==",
                };

                let lhs = self.gen_operand(lhs);
                let rhs = self.gen_operand(rhs);

                format!("{} {} {}", lhs, op, rhs)
            }
            Value::Cast(operand, ty) => {
                let operand = self.gen_operand(operand);
                let ty = self.gen_type(ty);

                format!("({}) {}", ty, operand)
            }
            Value::AddressOf(mutable, place) => {
                let place = self.gen_place(place);

                if *mutable {
                    format!("&{}", place)
                } else {
                    format!("const &{}", place)
                }
            }
            Value::Struct(fields) => {
                let ty = self.gen_type(&value.ty());

                let fields: Vec<_> = fields.iter().map(|field| self.gen_operand(field)).collect();

                format!("({}) {{ {} }}", ty, fields.join(", "))
            }
            Value::Union(variant, variants) => {
                let variant_ty = variant.ty();
                let index = variants.iter().position(|ty| ty == variant_ty).unwrap();

                let variant = self.gen_operand(variant);
                let ty = self.gen_type(&value.ty());

                format!("({}) {{ .{} = {} }}", ty, variant_name(index), variant)
            }
            Value::Sizeof(ty) => format!("sizeof({})", self.gen_type(ty)),
            Value::Intrinsic(name, args, _) => match *name {
                "alloc" => {
                    let size = self.gen_operand(&args[0]);
                    format!("malloc({})", size)
                }
                "realloc" => {
                    let ptr = self.gen_operand(&args[0]);
                    let size = self.gen_operand(&args[1]);
                    format!("realloc({}, {})", ptr, size)
                }
                "dealloc" => {
                    let ptr = self.gen_operand(&args[0]);
                    self.bodies += &format!("    free({});\n", ptr);
                    self.gen_value(&Value::VOID)
                }
                "memcopy" => {
                    let dst = self.gen_operand(&args[0]);
                    let src = self.gen_operand(&args[1]);
                    let size = self.gen_operand(&args[2]);
                    self.bodies += &format!("    memcpy({}, {}, {});\n", dst, src, size);
                    self.gen_value(&Value::VOID)
                }
                _ => unimplemented!(),
            },
        }
    }

    fn gen_statement(&mut self, statement: &Statement) -> String {
        match statement {
            Statement::Assign(place, value) => {
                let place = self.gen_place(place);
                let value = self.gen_value(value);

                format!("{} = {};", place, value)
            }
        }
    }

    fn gen_terminator(&mut self, terminator: &Terminator) -> String {
        match terminator {
            Terminator::Goto(block_id) => format!("goto {};", block_name(*block_id)),
            Terminator::Return(value) => {
                let value = self.gen_value(value);

                format!("return {};", value)
            }
            Terminator::Switch {
                discriminant: operand,
                default,
                cases,
            } => {
                let operand = self.gen_operand(operand);

                let mut code = String::new();

                code += "switch (";
                code += &operand;
                code += ") {\n";

                for (value, target) in cases {
                    code += "      case ";
                    code += &format!("{}", value);
                    code += ": goto ";
                    code += &block_name(*target);
                    code += ";\n";
                }

                code += "      default: goto ";
                code += &block_name(*default);
                code += ";\n";
                code += "    }";

                code
            }

            Terminator::Call {
                callee,
                arguments,
                destination,
                target,
            } => {
                let callee = self.gen_operand(callee);

                let mut args = String::new();

                for (i, argument) in arguments.iter().enumerate() {
                    if i > 0 {
                        args += ", ";
                    }

                    args += &self.gen_operand(argument);
                }

                let destination = self.gen_place(destination);

                match target {
                    Some(target) => format!(
                        "{} = {}({}); goto {};",
                        destination,
                        callee,
                        args,
                        block_name(*target)
                    ),
                    None => format!("{}({});", callee, args),
                }
            }
            Terminator::Unreachable => String::from("/* unreachable */"),
        }
    }

    fn gen_block(&mut self, block_id: BlockId, block: &Block) {
        self.bodies += "  ";
        self.bodies += &block_name(block_id);
        self.bodies += ":\n";

        for statement in &block.statements {
            let statement = self.gen_statement(statement);

            self.bodies += "    ";
            self.bodies += &statement;
            self.bodies += "\n";
        }

        let terminator = self.gen_terminator(&block.terminator);

        self.bodies += "    ";
        self.bodies += &terminator;
        self.bodies += "\n";
    }

    fn gen_body(&mut self, body_id: BodyId, body: &Body) {
        let name = body_name(body_id);
        let output = self.gen_type(&body.output);

        self.define += &format!("{} {}(", output, name);
        self.bodies += &format!("{} {}(", output, name);

        for (i, argument) in body.arguments.iter().enumerate() {
            let local = &body.locals[*argument];
            let ty = self.gen_type(&local.ty);
            let name = local_name(*argument);

            if i > 0 {
                self.define += ", ";
                self.bodies += ", ";
            }

            self.define += &format!("{} {}", ty, name);
            self.bodies += &format!("{} {}", ty, name);
        }

        self.define += ");\n";
        self.bodies += ") {\n";

        for (local, decl) in body.locals.iter() {
            if body.arguments.contains(&local) {
                continue;
            }

            let ty = self.gen_type(&decl.ty);
            let name = local_name(local);

            self.bodies += &format!("    {} {};\n", ty, name);
        }

        for (block_id, block) in body.blocks.iter() {
            self.bodies += "\n";
            self.gen_block(block_id, block);
        }

        self.bodies += "}\n\n";
    }
}

fn local_name(local: Local) -> String {
    format!("_{}", local.index)
}

fn field_name(index: usize) -> String {
    format!("_{}", index)
}

fn variant_name(index: usize) -> String {
    format!("_{}", index)
}

fn block_name(block_id: BlockId) -> String {
    format!("basic_block{}", block_id.index())
}

fn body_name(body_id: BodyId) -> String {
    format!("_rite_body_{}", body_id.index())
}

fn header() -> String {
    let mut code = String::new();

    code += "#include <stdint.h>\n";
    code += "#include <stddef.h>\n";
    code += "#include <stdlib.h>\n";
    code += "#include <string.h>\n";

    code
}

pub fn codegen(unit: &Unit) -> String {
    let mut codegen = Codegen {
        unit,
        define: String::new(),
        bodies: String::new(),
        structs: HashMap::new(),
        unions: HashMap::new(),
        functions: HashMap::new(),
    };

    for (body_id, body) in unit.bodies.iter() {
        codegen.gen_body(body_id, body);
    }

    let mut code = header();

    code += "\n";
    code += "/* ---------- Types ---------- */\n";
    code += &codegen.define;
    code += "\n";
    code += "/* ---------- Bodies --------- */\n";
    code += &codegen.bodies;

    if let Some(entry) = unit.entry {
        code += "\n";
        code += "int main(int argc, char** argv) {\n";
        code += "  return ";
        code += &body_name(entry);
        code += "((size_t) argc, (const uint8_t**) argv);\n";
        code += "}\n";
    }

    code
}
