use std::collections::HashMap;

use ritec_mir::{
    Block, BlockId, Body, BodyId, Const, ConstKind, Local, Operand, Place, ProjectionKind,
    Statement, Terminator, Type, Unit, Value,
};

struct Codegen<'a> {
    unit: &'a Unit,
    types: String,
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

                self.types += &format!("typedef {} (*{})({});\n", output, name, args);
                self.types += "\n";

                self.functions.insert(types, name.clone());

                name
            }
            Type::Struct { fields } => {
                if let Some(name) = self.structs.get(fields) {
                    return name.clone();
                }

                self.types += "typedef struct {\n";

                for (i, field) in fields.iter().enumerate() {
                    let ty = self.gen_type(field);

                    self.types += &format!("    {} {};\n", ty, field_name(i));
                }

                let name = format!("_rite_struct_{}", self.structs.len());

                self.types += &format!("}} {};\n", name);
                self.types += "\n";

                self.structs.insert(fields.clone(), name.clone());

                name
            }
            Type::Union { variants } => {
                if let Some(name) = self.unions.get(variants) {
                    return name.clone();
                }

                self.types += "typedef union {\n";

                for (i, variant) in variants.iter().enumerate() {
                    let ty = self.gen_type(variant);

                    self.types += &format!("    {} _{};\n", ty, i);
                }

                let name = format!("_rite_union_{}", self.unions.len());

                self.types += &format!("}} {};\n", name);
                self.types += "\n";

                self.unions.insert(variants.clone(), name.clone());

                name
            }
        }
    }

    fn gen_place(&mut self, place: &Place) -> String {
        let mut code = local_name(place.base);

        for projection in &place.projections {
            match projection.kind {
                ProjectionKind::Deref => todo!(),
                ProjectionKind::Field(index) => code = format!("{}.{}", code, field_name(index)),
            }
        }

        code
    }

    fn gen_const(&mut self, constant: &Const) -> String {
        match constant.kind {
            ConstKind::Int(v) => format!("{}", v),
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
            Value::Struct(fields) => {
                let ty = self.gen_type(&value.ty());

                let fields: Vec<_> = fields.iter().map(|field| self.gen_operand(field)).collect();

                format!("({}) {{ {} }}", ty, fields.join(", "))
            }
            Value::Intrinsic(name, args, _) => match *name {
                "malloc" => {
                    let size = self.gen_operand(&args[0]);
                    format!("malloc({})", size)
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
            Terminator::Unreachable => unimplemented!(),
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

        self.bodies += &format!("{} {}(", output, name);

        for (i, argument) in body.arguments.iter().enumerate() {
            let local = &body.locals[*argument];
            let ty = self.gen_type(&local.ty);
            let name = local_name(*argument);

            if i > 0 {
                self.bodies += ", ";
            }

            self.bodies += &format!("{} {}", ty, name);
        }

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

    code
}

pub fn codegen(unit: &Unit) -> String {
    let mut codegen = Codegen {
        unit,
        types: String::new(),
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
    code += &codegen.types;
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
