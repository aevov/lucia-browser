/// ACL 3.0 Interpreter for the Nara Runtime
/// Full grammar parser with tokenizer, AST, and evaluator.
/// Implements the Aevov Character Language spec within a native Rust environment.

use std::collections::HashMap;
use serde::{Deserialize, Serialize};

// --- Token Types ---

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // ACL operators (Unicode glyphs)
    Superposition,          // ◬
    Gate,                   // ⧈
    Entangle,               // ☥
    Measure,                // ⟓
    NeuralNetwork,          // Ψ

    // Language constructs
    Fn,                     // fn
    Let,                    // let
    Return,                 // return
    Circuit,                // circuit
    If,                     // if
    Else,                   // else

    // Literals & identifiers
    Ident(String),
    StringLit(String),
    NumberLit(f64),
    IntLit(i64),

    // Delimiters & operators
    LParen,                 // (
    RParen,                 // )
    LBrace,                 // {
    RBrace,                 // }
    LAngle,                 // <
    RAngle,                 // >
    Comma,
    Dot,
    Equals,                 // =
    Plus,
    Minus,
    Star,
    Slash,
    Semicolon,

    // Special
    Eof,
}

// --- Tokenizer ---

pub struct Tokenizer {
    chars: Vec<char>,
    pos: usize,
}

impl Tokenizer {
    pub fn new(source: &str) -> Self {
        Self {
            chars: source.chars().collect(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<char> {
        self.chars.get(self.pos).copied()
    }

    fn advance(&mut self) -> Option<char> {
        let ch = self.chars.get(self.pos).copied();
        self.pos += 1;
        ch
    }

    fn skip_whitespace(&mut self) {
        while let Some(ch) = self.peek() {
            if ch.is_whitespace() {
                self.advance();
            } else if ch == '/' && self.chars.get(self.pos + 1) == Some(&'/') {
                // Line comment
                while let Some(c) = self.advance() {
                    if c == '\n' {
                        break;
                    }
                }
            } else {
                break;
            }
        }
    }

    fn read_string(&mut self) -> String {
        let mut s = String::new();
        // Skip opening quote
        self.advance();
        while let Some(ch) = self.advance() {
            if ch == '"' {
                break;
            }
            s.push(ch);
        }
        s
    }

    fn read_number(&mut self) -> Token {
        let mut num_str = String::new();
        let mut is_float = false;
        while let Some(ch) = self.peek() {
            if ch.is_ascii_digit() {
                num_str.push(ch);
                self.advance();
            } else if ch == '.' && !is_float {
                is_float = true;
                num_str.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        if is_float {
            Token::NumberLit(num_str.parse().unwrap_or(0.0))
        } else {
            Token::IntLit(num_str.parse().unwrap_or(0))
        }
    }

    fn read_ident(&mut self) -> String {
        let mut s = String::new();
        while let Some(ch) = self.peek() {
            if ch.is_alphanumeric() || ch == '_' {
                s.push(ch);
                self.advance();
            } else {
                break;
            }
        }
        s
    }

    pub fn tokenize(&mut self) -> Vec<Token> {
        let mut tokens = Vec::new();

        loop {
            self.skip_whitespace();

            match self.peek() {
                None => {
                    tokens.push(Token::Eof);
                    break;
                }
                Some(ch) => {
                    let token = match ch {
                        // ACL glyphs
                        '◬' => { self.advance(); Token::Superposition }
                        '⧈' => { self.advance(); Token::Gate }
                        '☥' => { self.advance(); Token::Entangle }
                        '⟓' => { self.advance(); Token::Measure }
                        'Ψ' => { self.advance(); Token::NeuralNetwork }

                        // Delimiters
                        '(' => { self.advance(); Token::LParen }
                        ')' => { self.advance(); Token::RParen }
                        '{' => { self.advance(); Token::LBrace }
                        '}' => { self.advance(); Token::RBrace }
                        '<' => { self.advance(); Token::LAngle }
                        '>' => { self.advance(); Token::RAngle }
                        ',' => { self.advance(); Token::Comma }
                        '.' => { self.advance(); Token::Dot }
                        '=' => { self.advance(); Token::Equals }
                        '+' => { self.advance(); Token::Plus }
                        '-' => { self.advance(); Token::Minus }
                        '*' => { self.advance(); Token::Star }
                        ';' => { self.advance(); Token::Semicolon }

                        '"' => Token::StringLit(self.read_string()),

                        c if c.is_ascii_digit() => self.read_number(),

                        c if c.is_alphabetic() || c == '_' => {
                            let ident = self.read_ident();
                            match ident.as_str() {
                                "fn" => Token::Fn,
                                "let" => Token::Let,
                                "return" => Token::Return,
                                "circuit" => Token::Circuit,
                                "if" => Token::If,
                                "else" => Token::Else,
                                _ => Token::Ident(ident),
                            }
                        }

                        _ => {
                            // Skip unknown characters
                            self.advance();
                            continue;
                        }
                    };
                    tokens.push(token);
                }
            }
        }

        tokens
    }
}

// --- AST ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AstNode {
    /// Function definition: fn name(params) { body }
    FnDef {
        name: String,
        params: Vec<String>,
        body: Vec<AstNode>,
    },
    /// Circuit definition: circuit name<qubits>(depth) { body }
    CircuitDef {
        name: String,
        qubits: u8,
        depth: u8,
        body: Vec<AstNode>,
    },
    /// Variable binding: let name = expr
    LetBinding {
        name: String,
        value: Box<AstNode>,
    },
    /// Return statement
    Return(Box<AstNode>),
    /// Superposition: ◬<n> target
    Superposition {
        qubits: u8,
        target: String,
    },
    /// Gate application: ⧈ target "GATE_NAME" [param]
    GateApply {
        target: String,
        gate_name: String,
        param: Option<f64>,
    },
    /// Entanglement: ☥ target1 target2
    Entangle {
        target_a: String,
        target_b: String,
    },
    /// Measurement: ⟓ target
    Measure {
        target: String,
    },
    /// Neural network invoke: Ψ target "NET_NAME"
    NeuralInvoke {
        target: String,
        network_name: String,
    },
    /// Function call: name(args)
    FnCall {
        name: String,
        args: Vec<AstNode>,
    },
    /// Identifier reference
    Ident(String),
    /// String literal
    StringLit(String),
    /// Number literal
    NumberLit(f64),
    /// Integer literal
    IntLit(i64),
    /// If expression
    IfExpr {
        condition: Box<AstNode>,
        then_body: Vec<AstNode>,
        else_body: Option<Vec<AstNode>>,
    },
    /// No-op (comment, whitespace, etc.)
    Noop,
}

// --- Parser ---

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
}

impl Parser {
    pub fn new(tokens: Vec<Token>) -> Self {
        Self { tokens, pos: 0 }
    }

    fn peek(&self) -> &Token {
        self.tokens.get(self.pos).unwrap_or(&Token::Eof)
    }

    fn advance(&mut self) -> Token {
        let tok = self.tokens.get(self.pos).cloned().unwrap_or(Token::Eof);
        self.pos += 1;
        tok
    }

    fn expect_ident(&mut self) -> Result<String, String> {
        match self.advance() {
            Token::Ident(s) => Ok(s),
            other => Err(format!("Expected identifier, got {:?}", other)),
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<(), String> {
        let got = self.advance();
        if &got == expected {
            Ok(())
        } else {
            Err(format!("Expected {:?}, got {:?}", expected, got))
        }
    }

    pub fn parse_program(&mut self) -> Result<Vec<AstNode>, String> {
        let mut nodes = Vec::new();
        while *self.peek() != Token::Eof {
            let node = self.parse_statement()?;
            match node {
                AstNode::Noop => continue,
                _ => nodes.push(node),
            }
        }
        Ok(nodes)
    }

    fn parse_statement(&mut self) -> Result<AstNode, String> {
        match self.peek().clone() {
            Token::Fn => self.parse_fn_def(),
            Token::Circuit => self.parse_circuit_def(),
            Token::Let => self.parse_let_binding(),
            Token::Return => self.parse_return(),
            Token::If => self.parse_if(),
            Token::Superposition => self.parse_superposition(),
            Token::Gate => self.parse_gate(),
            Token::Entangle => self.parse_entangle(),
            Token::Measure => self.parse_measure(),
            Token::NeuralNetwork => self.parse_neural(),
            Token::Ident(_) => self.parse_fn_call_or_ident(),
            _ => {
                self.advance();
                Ok(AstNode::Noop)
            }
        }
    }

    fn parse_fn_def(&mut self) -> Result<AstNode, String> {
        self.advance(); // consume 'fn'
        let name = self.expect_ident()?;
        self.expect(&Token::LParen)?;

        let mut params = Vec::new();
        while *self.peek() != Token::RParen && *self.peek() != Token::Eof {
            params.push(self.expect_ident()?);
            if *self.peek() == Token::Comma {
                self.advance();
            }
        }
        self.expect(&Token::RParen)?;
        let body = self.parse_block()?;

        Ok(AstNode::FnDef { name, params, body })
    }

    fn parse_circuit_def(&mut self) -> Result<AstNode, String> {
        self.advance(); // consume 'circuit'
        let name = self.expect_ident()?;

        // Parse <qubits>
        let mut qubits = 3u8;
        if *self.peek() == Token::LAngle {
            self.advance();
            if let Token::IntLit(n) = self.advance() {
                qubits = n as u8;
            }
            self.expect(&Token::RAngle)?;
        }

        // Parse (depth)
        let mut depth = 1u8;
        if *self.peek() == Token::LParen {
            self.advance();
            if let Token::IntLit(n) = self.advance() {
                depth = n as u8;
            }
            self.expect(&Token::RParen)?;
        }

        let body = self.parse_block()?;
        Ok(AstNode::CircuitDef { name, qubits, depth, body })
    }

    fn parse_let_binding(&mut self) -> Result<AstNode, String> {
        self.advance(); // consume 'let'
        let name = self.expect_ident()?;
        self.expect(&Token::Equals)?;
        let value = self.parse_expression()?;
        Ok(AstNode::LetBinding {
            name,
            value: Box::new(value),
        })
    }

    fn parse_return(&mut self) -> Result<AstNode, String> {
        self.advance(); // consume 'return'
        let value = self.parse_expression()?;
        Ok(AstNode::Return(Box::new(value)))
    }

    fn parse_if(&mut self) -> Result<AstNode, String> {
        self.advance(); // consume 'if'
        let condition = self.parse_expression()?;
        let then_body = self.parse_block()?;
        let else_body = if *self.peek() == Token::Else {
            self.advance();
            Some(self.parse_block()?)
        } else {
            None
        };
        Ok(AstNode::IfExpr {
            condition: Box::new(condition),
            then_body,
            else_body,
        })
    }

    fn parse_superposition(&mut self) -> Result<AstNode, String> {
        self.advance(); // consume ◬
        let mut qubits = 3u8;
        if *self.peek() == Token::LAngle {
            self.advance();
            if let Token::IntLit(n) = self.advance() {
                qubits = n as u8;
            }
            self.expect(&Token::RAngle)?;
        }
        let target = self.expect_ident()?;
        Ok(AstNode::Superposition { qubits, target })
    }

    fn parse_gate(&mut self) -> Result<AstNode, String> {
        self.advance(); // consume ⧈
        let target = self.expect_ident()?;
        let gate_name = match self.peek().clone() {
            Token::StringLit(s) => { self.advance(); s }
            _ => "IDENTITY".to_string(),
        };
        let param = match self.peek() {
            Token::NumberLit(_) | Token::IntLit(_) => {
                match self.advance() {
                    Token::NumberLit(n) => Some(n),
                    Token::IntLit(n) => Some(n as f64),
                    _ => None,
                }
            }
            Token::Ident(_) => { self.advance(); None }
            _ => None,
        };
        Ok(AstNode::GateApply { target, gate_name, param })
    }

    fn parse_entangle(&mut self) -> Result<AstNode, String> {
        self.advance(); // consume ☥
        let target_a = self.expect_ident()?;
        let target_b = match self.peek().clone() {
            Token::StringLit(s) => { self.advance(); s }
            Token::Ident(_) => self.expect_ident()?,
            _ => "default".to_string(),
        };
        Ok(AstNode::Entangle { target_a, target_b })
    }

    fn parse_measure(&mut self) -> Result<AstNode, String> {
        self.advance(); // consume ⟓
        let target = self.expect_ident()?;
        Ok(AstNode::Measure { target })
    }

    fn parse_neural(&mut self) -> Result<AstNode, String> {
        self.advance(); // consume Ψ
        let target = self.expect_ident()?;
        let network_name = match self.peek().clone() {
            Token::StringLit(s) => { self.advance(); s }
            _ => "default".to_string(),
        };
        Ok(AstNode::NeuralInvoke { target, network_name })
    }

    fn parse_fn_call_or_ident(&mut self) -> Result<AstNode, String> {
        let name = self.expect_ident()?;
        if *self.peek() == Token::LParen {
            self.advance(); // consume (
            let mut args = Vec::new();
            while *self.peek() != Token::RParen && *self.peek() != Token::Eof {
                args.push(self.parse_expression()?);
                if *self.peek() == Token::Comma {
                    self.advance();
                }
            }
            self.expect(&Token::RParen)?;
            Ok(AstNode::FnCall { name, args })
        } else {
            Ok(AstNode::Ident(name))
        }
    }

    fn parse_expression(&mut self) -> Result<AstNode, String> {
        match self.peek().clone() {
            Token::StringLit(s) => { self.advance(); Ok(AstNode::StringLit(s)) }
            Token::NumberLit(n) => { self.advance(); Ok(AstNode::NumberLit(n)) }
            Token::IntLit(n) => { self.advance(); Ok(AstNode::IntLit(n)) }
            Token::Ident(_) => self.parse_fn_call_or_ident(),
            Token::Superposition => self.parse_superposition(),
            Token::Gate => self.parse_gate(),
            Token::Entangle => self.parse_entangle(),
            Token::Measure => self.parse_measure(),
            Token::NeuralNetwork => self.parse_neural(),
            _ => {
                self.advance();
                Ok(AstNode::Noop)
            }
        }
    }

    fn parse_block(&mut self) -> Result<Vec<AstNode>, String> {
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        while *self.peek() != Token::RBrace && *self.peek() != Token::Eof {
            let node = self.parse_statement()?;
            match node {
                AstNode::Noop => continue,
                _ => stmts.push(node),
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(stmts)
    }
}

// --- Evaluator ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Value {
    Nil,
    Int(i64),
    Float(f64),
    Str(String),
    QuditState { qubits: u8, amplitudes: Vec<f64> },
    NeuralOutput { activations: Vec<f64> },
    MeasureResult { collapsed: i64, probability: f64 },
}

pub struct AclInterpreter {
    variables: HashMap<String, Value>,
    functions: HashMap<String, AstNode>,
    circuits: HashMap<String, AstNode>,
    draw_commands: Vec<DrawCommandOutput>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrawCommandOutput {
    pub op: String,
    pub target: String,
    pub params: HashMap<String, String>,
}

impl AclInterpreter {
    pub fn new() -> Self {
        Self {
            variables: HashMap::new(),
            functions: HashMap::new(),
            circuits: HashMap::new(),
            draw_commands: Vec::new(),
        }
    }

    /// Parse ACL source into AST
    pub fn parse(&self, source: &str) -> Vec<AstNode> {
        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer.tokenize();
        let mut parser = Parser::new(tokens);
        parser.parse_program().unwrap_or_default()
    }

    /// Execute an AST program
    pub fn execute(&mut self, program: &[AstNode]) -> Result<Value, String> {
        let mut last_value = Value::Nil;

        // First pass: register functions and circuits
        for node in program {
            match node {
                AstNode::FnDef { name, .. } => {
                    self.functions.insert(name.clone(), node.clone());
                }
                AstNode::CircuitDef { name, .. } => {
                    self.circuits.insert(name.clone(), node.clone());
                }
                _ => {}
            }
        }

        // Second pass: execute top-level statements
        for node in program {
            match node {
                AstNode::FnDef { .. } | AstNode::CircuitDef { .. } => {
                    // Already registered
                }
                _ => {
                    last_value = self.eval_node(node)?;
                }
            }
        }

        Ok(last_value)
    }

    fn eval_node(&mut self, node: &AstNode) -> Result<Value, String> {
        match node {
            AstNode::Superposition { qubits, target } => {
                let n = *qubits as usize;
                let dim = 1 << n; // 2^n
                let amp = 1.0 / (dim as f64).sqrt();
                let amplitudes = vec![amp; dim];
                let val = Value::QuditState {
                    qubits: *qubits,
                    amplitudes,
                };
                self.variables.insert(target.clone(), val.clone());
                self.emit_draw("superposition", target, &[("qubits", &qubits.to_string())]);
                Ok(val)
            }

            AstNode::GateApply { target, gate_name, param } => {
                // Apply gate transformation to qudit state
                if let Some(Value::QuditState { qubits, mut amplitudes }) =
                    self.variables.get(target).cloned()
                {
                    match gate_name.as_str() {
                        "HADAMARD" => {
                            // Simplified Hadamard: redistribute amplitudes
                            let n = amplitudes.len();
                            let factor = 1.0 / (n as f64).sqrt();
                            amplitudes = vec![factor; n];
                        }
                        "PHASE" => {
                            let angle = param.unwrap_or(std::f64::consts::FRAC_PI_4);
                            for (i, a) in amplitudes.iter_mut().enumerate() {
                                if i % 2 == 1 {
                                    *a *= angle.cos();
                                }
                            }
                        }
                        _ => {}
                    }
                    let val = Value::QuditState { qubits, amplitudes };
                    self.variables.insert(target.clone(), val.clone());
                    self.emit_draw("gate", target, &[("gate", gate_name)]);
                    Ok(val)
                } else {
                    self.emit_draw("gate", target, &[("gate", gate_name)]);
                    Ok(Value::Nil)
                }
            }

            AstNode::Entangle { target_a, target_b } => {
                self.emit_draw("entangle", target_a, &[("partner", target_b)]);
                Ok(Value::Nil)
            }

            AstNode::Measure { target } => {
                if let Some(Value::QuditState { amplitudes, .. }) =
                    self.variables.get(target).cloned()
                {
                    // Simplified measurement: pick state with highest amplitude
                    let max_idx = amplitudes
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    let prob = amplitudes.get(max_idx).copied().unwrap_or(0.0).powi(2);
                    let result = Value::MeasureResult {
                        collapsed: max_idx as i64,
                        probability: prob,
                    };
                    self.variables.insert(target.clone(), result.clone());
                    self.emit_draw("measure", target, &[("result", &max_idx.to_string())]);
                    Ok(result)
                } else {
                    Ok(Value::MeasureResult {
                        collapsed: 0,
                        probability: 1.0,
                    })
                }
            }

            AstNode::NeuralInvoke { target, network_name } => {
                // Emit neural activation with resonance pattern
                let activations = vec![0.8, 0.6, 0.9, 0.7]; // Simulated
                let val = Value::NeuralOutput { activations };
                self.variables.insert(target.clone(), val.clone());
                self.emit_draw("neural", target, &[("network", network_name)]);
                Ok(val)
            }

            AstNode::LetBinding { name, value } => {
                let val = self.eval_node(value)?;
                self.variables.insert(name.clone(), val.clone());
                Ok(val)
            }

            AstNode::Return(expr) => self.eval_node(expr),

            AstNode::FnCall { name, args } => {
                let mut arg_values = Vec::new();
                for arg in args {
                    arg_values.push(self.eval_node(arg)?);
                }

                if let Some(func) = self.functions.get(name).cloned() {
                    if let AstNode::FnDef { params, body, .. } = func {
                        // Bind arguments to parameters
                        for (param, val) in params.iter().zip(arg_values.iter()) {
                            self.variables.insert(param.clone(), val.clone());
                        }
                        let mut result = Value::Nil;
                        for stmt in &body {
                            result = self.eval_node(stmt)?;
                        }
                        Ok(result)
                    } else {
                        Ok(Value::Nil)
                    }
                } else {
                    // Built-in function calls
                    match name.as_str() {
                        "anyon_compress" => Ok(Value::Str("compressed_state".to_string())),
                        "ARS_4_0_sync" => Ok(Value::Str("sync_lock_active".to_string())),
                        _ => Ok(Value::Nil),
                    }
                }
            }

            AstNode::Ident(name) => {
                Ok(self.variables.get(name).cloned().unwrap_or(Value::Nil))
            }

            AstNode::StringLit(s) => Ok(Value::Str(s.clone())),
            AstNode::NumberLit(n) => Ok(Value::Float(*n)),
            AstNode::IntLit(n) => Ok(Value::Int(*n)),

            AstNode::IfExpr { condition, then_body, else_body } => {
                let cond = self.eval_node(condition)?;
                let truthy = match cond {
                    Value::Nil => false,
                    Value::Int(0) => false,
                    Value::Float(f) if f == 0.0 => false,
                    Value::Str(ref s) if s.is_empty() => false,
                    _ => true,
                };
                let body = if truthy { then_body } else {
                    match else_body {
                        Some(b) => b,
                        None => return Ok(Value::Nil),
                    }
                };
                let mut result = Value::Nil;
                for stmt in body {
                    result = self.eval_node(stmt)?;
                }
                Ok(result)
            }

            AstNode::FnDef { .. } | AstNode::CircuitDef { .. } => Ok(Value::Nil),
            AstNode::Noop => Ok(Value::Nil),
        }
    }

    fn emit_draw(&mut self, op: &str, target: &str, extra: &[(&str, &str)]) {
        let mut params = HashMap::new();
        for (k, v) in extra {
            params.insert(k.to_string(), v.to_string());
        }
        self.draw_commands.push(DrawCommandOutput {
            op: op.to_string(),
            target: target.to_string(),
            params,
        });
    }

    /// Get draw commands emitted during execution
    pub fn take_draw_commands(&mut self) -> Vec<DrawCommandOutput> {
        std::mem::take(&mut self.draw_commands)
    }

    /// Get current variable state
    pub fn get_variables(&self) -> &HashMap<String, Value> {
        &self.variables
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_acl_glyphs() {
        let mut tok = Tokenizer::new("◬<3> state\n⧈ state \"HADAMARD\"");
        let tokens = tok.tokenize();
        assert_eq!(tokens[0], Token::Superposition);
        assert_eq!(tokens[1], Token::LAngle);
    }

    #[test]
    fn test_parse_superposition() {
        let source = "◬<3> task_state";
        let interp = AclInterpreter::new();
        let ast = interp.parse(source);
        assert!(!ast.is_empty());
        match &ast[0] {
            AstNode::Superposition { qubits, target } => {
                assert_eq!(*qubits, 3);
                assert_eq!(target, "task_state");
            }
            _ => panic!("Expected Superposition"),
        }
    }

    #[test]
    fn test_full_acl_execution() {
        let source = r#"
            ◬<3> task_state
            ⧈ task_state "HADAMARD"
            Ψ nara_brain "GLOBAL_AWARENESS"
            ⟓ task_state
        "#;
        let mut interp = AclInterpreter::new();
        let ast = interp.parse(source);
        let result = interp.execute(&ast);
        assert!(result.is_ok());
        assert_eq!(interp.take_draw_commands().len(), 4);
    }

    #[test]
    fn test_parse_architecture_acl() {
        let source = r#"
            fn orchestrate_nara_mesh(node_id, task_load) {
                Ψ nara_brain "GLOBAL_AWARENESS"
                ◬<3> task_state
                ☥ node_id "UOE_ROOT"
            }

            fn main() {
                orchestrate_nara_mesh("nara-local-0", 0.75)
            }
        "#;
        let mut interp = AclInterpreter::new();
        let ast = interp.parse(source);
        assert!(ast.len() >= 2);
        let result = interp.execute(&ast);
        assert!(result.is_ok());
    }
}
