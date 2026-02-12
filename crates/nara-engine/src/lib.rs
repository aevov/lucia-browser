use std::sync::Arc;
use std::collections::HashMap;
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use quantumfs::QuantumFS;
use acl_interpreter::{AclInterpreter, AstNode};

/// Nara Runtime - The ACL-native logic engine for cr8OS
/// Manages script execution, UOE optimization, and draw queue for the painter.
pub struct NaraRuntime {
    qfs: Arc<RwLock<QuantumFS>>,
    draw_queue: Arc<RwLock<Vec<DrawCommand>>>,
    interpreter: AclInterpreter,
    uoe: UniversalOptimizationEngine,
    execution_log: Vec<ExecutionRecord>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum DrawCommand {
    CreateCard { id: String, x: f32, y: f32, color: [f32; 3], resonance: f32 },
    UpdateCard { id: String, x: f32, y: f32, resonance: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionRecord {
    pub script_hash: u64,
    pub ops_count: usize,
    pub draw_commands_emitted: usize,
    pub optimization_applied: bool,
    pub elapsed_ms: f64,
}

/// Universal Optimization Engine - MetaCognitive Orchestrator
/// Analyzes ACL scripts and predicts resource requirements.
pub struct UniversalOptimizationEngine {
    /// Cache of previously optimized AST trees keyed by script hash
    ast_cache: HashMap<u64, Vec<AstNode>>,
    /// Predicted resource needs per script pattern
    resource_predictions: HashMap<String, ResourcePrediction>,
    /// Total optimization passes performed
    passes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourcePrediction {
    pub estimated_qubits: u32,
    pub estimated_gates: u32,
    pub estimated_measurements: u32,
    pub neural_invocations: u32,
    pub estimated_draw_commands: u32,
    pub confidence: f64,
}

impl UniversalOptimizationEngine {
    pub fn new() -> Self {
        Self {
            ast_cache: HashMap::new(),
            resource_predictions: HashMap::new(),
            passes: 0,
        }
    }

    /// Analyze an AST and predict resource requirements
    pub fn analyze(&mut self, ast: &[AstNode]) -> ResourcePrediction {
        let mut pred = ResourcePrediction {
            estimated_qubits: 0,
            estimated_gates: 0,
            estimated_measurements: 0,
            neural_invocations: 0,
            estimated_draw_commands: 0,
            confidence: 0.0,
        };

        self.count_ops(ast, &mut pred);

        // Confidence based on analysis depth
        pred.estimated_draw_commands = pred.estimated_qubits
            + pred.estimated_gates
            + pred.estimated_measurements
            + pred.neural_invocations;
        pred.confidence = if pred.estimated_draw_commands > 0 {
            0.85
        } else {
            0.5
        };

        pred
    }

    fn count_ops(&self, nodes: &[AstNode], pred: &mut ResourcePrediction) {
        for node in nodes {
            match node {
                AstNode::Superposition { qubits, .. } => {
                    pred.estimated_qubits += *qubits as u32;
                }
                AstNode::GateApply { .. } => {
                    pred.estimated_gates += 1;
                }
                AstNode::Measure { .. } => {
                    pred.estimated_measurements += 1;
                }
                AstNode::NeuralInvoke { .. } => {
                    pred.neural_invocations += 1;
                }
                AstNode::FnDef { body, .. } | AstNode::CircuitDef { body, .. } => {
                    self.count_ops(body, pred);
                }
                AstNode::IfExpr { then_body, else_body, .. } => {
                    self.count_ops(then_body, pred);
                    if let Some(eb) = else_body {
                        self.count_ops(eb, pred);
                    }
                }
                _ => {}
            }
        }
    }

    /// Optimize an AST - gate fusion, dead code elimination, and reordering
    pub fn optimize(&mut self, ast: Vec<AstNode>, script_hash: u64) -> Vec<AstNode> {
        self.passes += 1;

        // Check cache
        if let Some(cached) = self.ast_cache.get(&script_hash) {
            log::info!("[UOE] Cache hit for script {:016x}", script_hash);
            return cached.clone();
        }

        // Optimization pass 1: Gate fusion
        // Adjacent gates on the same target can be combined
        let optimized = self.fuse_gates(ast);

        // Optimization pass 2: Dead qudit elimination
        // Remove superpositions that are never measured
        let optimized = self.eliminate_dead_qudits(optimized);

        // Cache result
        self.ast_cache.insert(script_hash, optimized.clone());

        log::info!(
            "[UOE] Optimization pass #{} complete for {:016x}",
            self.passes,
            script_hash
        );
        optimized
    }

    fn fuse_gates(&self, nodes: Vec<AstNode>) -> Vec<AstNode> {
        let mut result = Vec::new();
        let mut i = 0;
        while i < nodes.len() {
            if i + 1 < nodes.len() {
                // Check for adjacent gates on same target
                if let (
                    AstNode::GateApply { target: t1, gate_name: g1, param: p1 },
                    AstNode::GateApply { target: t2, gate_name: g2, param: p2 },
                ) = (&nodes[i], &nodes[i + 1])
                {
                    if t1 == t2 && g1 == g2 {
                        // Fuse: combine parameters
                        let fused_param = match (p1, p2) {
                            (Some(a), Some(b)) => Some(a + b),
                            (Some(a), None) => Some(*a),
                            (None, Some(b)) => Some(*b),
                            _ => None,
                        };
                        result.push(AstNode::GateApply {
                            target: t1.clone(),
                            gate_name: format!("{}_FUSED", g1),
                            param: fused_param,
                        });
                        i += 2;
                        continue;
                    }
                }
            }
            result.push(nodes[i].clone());
            i += 1;
        }
        result
    }

    fn eliminate_dead_qudits(&self, nodes: Vec<AstNode>) -> Vec<AstNode> {
        // Collect all measured targets
        let mut measured: std::collections::HashSet<String> = std::collections::HashSet::new();
        for node in &nodes {
            if let AstNode::Measure { target } = node {
                measured.insert(target.clone());
            }
        }

        // Keep superpositions that are eventually measured, or keep all if no measurements
        if measured.is_empty() {
            return nodes;
        }

        nodes
            .into_iter()
            .filter(|node| {
                if let AstNode::Superposition { target, .. } = node {
                    measured.contains(target)
                } else {
                    true
                }
            })
            .collect()
    }

    /// Predict what the next ACL intent will be (strategic planning)
    pub fn predict_next(&self, current_ops: &[AstNode]) -> Vec<String> {
        let mut predictions = Vec::new();
        let mut has_superposition = false;
        let mut has_gate = false;

        for op in current_ops {
            match op {
                AstNode::Superposition { .. } => has_superposition = true,
                AstNode::GateApply { .. } => has_gate = true,
                _ => {}
            }
        }

        // Heuristic: if we have superpositions, gates likely follow
        if has_superposition && !has_gate {
            predictions.push("gate_application".to_string());
        }
        // If we have gates, measurement likely follows
        if has_gate {
            predictions.push("measurement".to_string());
        }
        // Neural invocations often follow measurements
        predictions.push("neural_invoke".to_string());

        predictions
    }

    pub fn get_stats(&self) -> UoeStats {
        UoeStats {
            total_passes: self.passes,
            cached_scripts: self.ast_cache.len(),
            prediction_count: self.resource_predictions.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UoeStats {
    pub total_passes: u64,
    pub cached_scripts: usize,
    pub prediction_count: usize,
}

// Simple hash function for script content
fn hash_script(script: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for byte in script.bytes() {
        h ^= byte as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl NaraRuntime {
    pub fn new(qfs: Arc<RwLock<QuantumFS>>) -> Self {
        Self {
            qfs,
            draw_queue: Arc::new(RwLock::new(Vec::new())),
            interpreter: AclInterpreter::new(),
            uoe: UniversalOptimizationEngine::new(),
            execution_log: Vec::new(),
        }
    }

    pub fn get_draw_queue(&self) -> Arc<RwLock<Vec<DrawCommand>>> {
        self.draw_queue.clone()
    }

    /// Execute an ACL 3.0 script with full parse -> optimize -> evaluate pipeline
    pub async fn execute_acl(&mut self, script: &str) -> Result<String, String> {
        let start = std::time::Instant::now();
        log::info!("[Nara] Executing ACL script: {} bytes", script.len());

        let script_hash = hash_script(script);

        // 1. Parse via full grammar parser
        let ast = self.interpreter.parse(script);
        if ast.is_empty() {
            return Ok("ACL: Empty program".to_string());
        }

        // 2. UOE: Analyze resource requirements
        let prediction = self.uoe.analyze(&ast);
        log::info!(
            "[UOE] Resource prediction: {} qubits, {} gates, {} measurements, {} neural (confidence: {:.0}%)",
            prediction.estimated_qubits,
            prediction.estimated_gates,
            prediction.estimated_measurements,
            prediction.neural_invocations,
            prediction.confidence * 100.0
        );

        // 3. UOE: Predict next intent for pre-warming
        let next_predictions = self.uoe.predict_next(&ast);
        if !next_predictions.is_empty() {
            log::info!("[UOE] Strategic planning - predicted next: {:?}", next_predictions);
        }

        // 4. UOE: Optimize AST
        let optimized_ast = self.uoe.optimize(ast, script_hash);

        // 5. Execute via interpreter
        let _result = self.interpreter.execute(&optimized_ast)
            .map_err(|e| format!("ACL execution error: {}", e))?;

        // 6. Convert interpreter draw commands to painter draw commands
        let draw_cmds = self.interpreter.take_draw_commands();
        let draw_count = draw_cmds.len();

        {
            let mut queue = self.draw_queue.write().await;
            for (i, cmd) in draw_cmds.iter().enumerate() {
                // Map interpreter draw ops to painter DrawCommands
                let card_cmd = match cmd.op.as_str() {
                    "superposition" => DrawCommand::CreateCard {
                        id: format!("acl-{}-{}", cmd.target, i),
                        x: -0.6 + (i as f32 * 0.35),
                        y: 0.3,
                        color: [0.0, 0.8, 1.0], // Cyan for superposition
                        resonance: 0.9,
                    },
                    "gate" => DrawCommand::CreateCard {
                        id: format!("acl-{}-{}", cmd.target, i),
                        x: -0.6 + (i as f32 * 0.35),
                        y: 0.0,
                        color: [0.8, 0.0, 1.0], // Purple for gates
                        resonance: 0.7,
                    },
                    "measure" => DrawCommand::CreateCard {
                        id: format!("acl-{}-{}", cmd.target, i),
                        x: -0.6 + (i as f32 * 0.35),
                        y: -0.3,
                        color: [1.0, 0.8, 0.0], // Gold for measurement
                        resonance: 1.0,
                    },
                    "neural" => DrawCommand::CreateCard {
                        id: format!("acl-{}-{}", cmd.target, i),
                        x: -0.6 + (i as f32 * 0.35),
                        y: -0.6,
                        color: [0.0, 1.0, 0.5], // Green for neural
                        resonance: 0.85,
                    },
                    "entangle" => DrawCommand::CreateCard {
                        id: format!("acl-{}-{}", cmd.target, i),
                        x: -0.6 + (i as f32 * 0.35),
                        y: 0.6,
                        color: [1.0, 0.0, 0.5], // Pink for entanglement
                        resonance: 0.95,
                    },
                    _ => continue,
                };
                queue.push(card_cmd);
            }
        }

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        // 7. Log execution record
        self.execution_log.push(ExecutionRecord {
            script_hash,
            ops_count: optimized_ast.len(),
            draw_commands_emitted: draw_count,
            optimization_applied: true,
            elapsed_ms: elapsed,
        });

        let result_str = format!(
            "ACL OK: {} ops, {} draw commands, {:.1}ms (UOE pass #{})",
            optimized_ast.len(),
            draw_count,
            elapsed,
            self.uoe.get_stats().total_passes
        );

        log::info!("[Nara] {}", result_str);
        Ok(result_str)
    }

    /// Get UOE optimization statistics
    pub fn get_uoe_stats(&self) -> UoeStats {
        self.uoe.get_stats()
    }

    /// Get execution history
    pub fn get_execution_log(&self) -> &[ExecutionRecord] {
        &self.execution_log
    }
}
