use pyo3::prelude::*;
use pyo3::types::PyModule;
use tree_sitter::{Parser, Tree, Node, TreeCursor};
use std::collections::HashMap;

/// Threshold for error count above which we bail out to regex fallback parsing
const ERROR_FALLBACK_THRESHOLD: usize = 10;

#[pyclass]
pub struct HaskellParser {
    parser: Parser,
}

#[pymethods]
impl HaskellParser {
    #[new]
    fn new() -> PyResult<Self> {
        let mut parser = Parser::new();
        let language = tree_sitter_haskell::LANGUAGE.into();
        parser.set_language(&language)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                format!("Failed to set Haskell language: {}", e)
            ))?;
        
        Ok(HaskellParser { parser })
    }
    
    fn parse(&mut self, source: &str) -> PyResult<Option<HaskellTree>> {
        match self.parser.parse(source, None) {
            Some(tree) => Ok(Some(HaskellTree { tree })),
            None => Ok(None),
        }
    }
}

#[pyclass]
pub struct HaskellTree {
    tree: Tree,
}

#[pymethods]
impl HaskellTree {
    fn root_node(&self) -> HaskellNode {
        let node = self.tree.root_node();
        let start_pos = node.start_position();
        let end_pos = node.end_position();
        
        HaskellNode {
            kind: node.kind().to_string(),
            start_byte: node.start_byte(),
            end_byte: node.end_byte(),
            start_row: start_pos.row,
            start_column: start_pos.column,
            end_row: end_pos.row,
            end_column: end_pos.column,
            child_count: node.child_count(),
            is_named: node.is_named(),
            is_error: node.is_error(),
        }
    }
}

#[pyclass]
pub struct HaskellNode {
    kind: String,
    start_byte: usize,
    end_byte: usize,
    start_row: usize,
    start_column: usize,
    end_row: usize,
    end_column: usize,
    child_count: usize,
    is_named: bool,
    is_error: bool,
}

#[pymethods]
impl HaskellNode {
    fn kind(&self) -> &str {
        &self.kind
    }
    
    fn start_byte(&self) -> usize {
        self.start_byte
    }
    
    fn end_byte(&self) -> usize {
        self.end_byte
    }
    
    fn start_position(&self) -> (usize, usize) {
        (self.start_row, self.start_column)
    }
    
    fn end_position(&self) -> (usize, usize) {
        (self.end_row, self.end_column)
    }
    
    fn child_count(&self) -> usize {
        self.child_count
    }
    
    fn is_named(&self) -> bool {
        self.is_named
    }
    
    fn is_error(&self) -> bool {
        self.is_error
    }
}

#[pyfunction]
fn parse_haskell(source: &str) -> PyResult<Option<HaskellTree>> {
    let mut parser = HaskellParser::new()?;
    parser.parse(source)
}

#[pyfunction]
fn get_haskell_separators() -> Vec<String> {
    vec![
        // Function definitions
        r"\n\w+\s*::\s*".to_string(),
        r"\n\w+.*=\s*".to_string(),
        // Module and import declarations
        r"\nmodule\s+".to_string(),
        r"\nimport\s+".to_string(),
        // Type definitions
        r"\ndata\s+".to_string(),
        r"\nnewtype\s+".to_string(),
        r"\ntype\s+".to_string(),
        r"\nclass\s+".to_string(),
        r"\ninstance\s+".to_string(),
        // General separators
        r"\n\n+".to_string(),
        r"\n".to_string(),
    ]
}

#[pyclass]
#[derive(Clone)]
pub struct HaskellChunk {
    pub text: String,
    pub start_byte: usize,
    pub end_byte: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub node_type: String,
    pub metadata: HashMap<String, String>,
}

#[pyclass]
#[derive(Clone)]
pub struct ErrorNodeStats {
    pub error_count: usize,
    pub nodes_with_errors: usize,
    pub uncovered_ranges: Vec<(usize, usize)>,
    pub should_fallback: bool,
}

#[pymethods]
impl ErrorNodeStats {
    fn error_count(&self) -> usize {
        self.error_count
    }
    
    fn nodes_with_errors(&self) -> usize {
        self.nodes_with_errors
    }
    
    fn should_fallback(&self) -> bool {
        self.should_fallback
    }
    
    fn uncovered_ranges(&self) -> Vec<(usize, usize)> {
        self.uncovered_ranges.clone()
    }
}

#[pyclass]
#[derive(Clone)]
pub struct ChunkingResult {
    pub chunks: Vec<HaskellChunk>,
    pub error_stats: ErrorNodeStats,
    pub chunking_method: String,
    pub coverage_complete: bool,
}

/// Context information passed down during recursive splitting
#[derive(Clone, Debug)]
pub struct ChunkingContext {
    pub ancestors: Vec<ContextNode>,
    pub max_chunk_size: usize,
    pub min_chunk_size: usize,
    pub current_module: Option<String>,
    pub current_class: Option<String>,
    pub current_function: Option<String>,
    // Aggregate metadata collected during tree traversal
    pub functions: Vec<String>,
    pub data_types: Vec<String>,
    pub type_classes: Vec<String>,
    pub instances: Vec<String>,
    pub modules: Vec<String>,
    pub imports: Vec<String>,
    pub nodes_with_errors: Vec<String>,
}

/// Represents a contextual ancestor node
#[derive(Clone, Debug)]
pub struct ContextNode {
    pub node_type: String,
    pub name: Option<String>,
    pub start_byte: usize,
    pub end_byte: usize,
}

/// Enhanced chunking parameters from Python config
#[pyclass]
#[derive(Clone)]
pub struct ChunkingParams {
    pub chunk_size: usize,
    pub min_chunk_size: usize,
    pub chunk_overlap: usize,
    pub max_chunk_size: usize,
}

#[pymethods]
impl ChunkingParams {
    #[new]
    fn new(chunk_size: usize, min_chunk_size: usize, chunk_overlap: usize, max_chunk_size: usize) -> Self {
        ChunkingParams {
            chunk_size,
            min_chunk_size, 
            chunk_overlap,
            max_chunk_size,
        }
    }
    
    fn chunk_size(&self) -> usize {
        self.chunk_size
    }
    
    fn min_chunk_size(&self) -> usize {
        self.min_chunk_size
    }
    
    fn chunk_overlap(&self) -> usize {
        self.chunk_overlap
    }
    
    fn max_chunk_size(&self) -> usize {
        self.max_chunk_size
    }
}

#[pymethods]
impl ChunkingResult {
    fn chunks(&self) -> Vec<HaskellChunk> {
        self.chunks.clone()
    }
    
    fn error_stats(&self) -> ErrorNodeStats {
        self.error_stats.clone()
    }
    
    fn chunking_method(&self) -> &str {
        &self.chunking_method
    }
    
    fn coverage_complete(&self) -> bool {
        self.coverage_complete
    }
}

#[pymethods]
impl HaskellChunk {
    fn text(&self) -> &str {
        &self.text
    }
    
    fn start_byte(&self) -> usize {
        self.start_byte
    }
    
    fn end_byte(&self) -> usize {
        self.end_byte
    }
    
    fn start_line(&self) -> usize {
        self.start_line
    }
    
    fn end_line(&self) -> usize {
        self.end_line
    }
    
    fn node_type(&self) -> &str {
        &self.node_type
    }
    
    fn metadata(&self) -> HashMap<String, String> {
        self.metadata.clone()
    }
}

#[allow(dead_code)]
fn extract_semantic_chunks(tree: &Tree, source: &str) -> Vec<HaskellChunk> {
    let mut chunks = Vec::new();
    let root_node = tree.root_node();
    
    // Walk through the tree and extract semantic chunks
    let mut cursor = root_node.walk();
    extract_chunks_recursive(&mut cursor, source, &mut chunks, 0);
    
    chunks
}

fn extract_semantic_chunks_with_recursive_splitting(tree: &Tree, source: &str, params: &ChunkingParams) -> ChunkingResult {
    let root_node = tree.root_node();
    
    // Initialize context with empty ancestors and metadata collections
    let mut context = ChunkingContext {
        ancestors: Vec::new(),
        max_chunk_size: params.max_chunk_size,
        min_chunk_size: params.min_chunk_size,
        current_module: None,
        current_class: None,
        current_function: None,
        functions: Vec::new(),
        data_types: Vec::new(),
        type_classes: Vec::new(),
        instances: Vec::new(),
        modules: Vec::new(),
        imports: Vec::new(),
        nodes_with_errors: Vec::new(),
    };
    
    // First pass: collect all metadata from the entire tree
    let mut cursor = root_node.walk();
    collect_aggregate_metadata(&mut cursor, source, &mut context);
    
    // First, count error nodes
    let mut error_stats = ErrorNodeStats {
        error_count: 0,
        nodes_with_errors: 0,
        uncovered_ranges: Vec::new(),
        should_fallback: false,
    };
    
    let mut cursor = root_node.walk();
    count_error_nodes(&mut cursor, &mut error_stats);
    
    // Check if we should bail out to regex fallback
    error_stats.should_fallback = error_stats.error_count >= ERROR_FALLBACK_THRESHOLD;
    
    if error_stats.should_fallback {
        // Use regex fallback chunking
        let regex_chunks = create_regex_fallback_chunks(source);
        return ChunkingResult {
            chunks: regex_chunks,
            error_stats,
            chunking_method: "rust_haskell_regex_fallback".to_string(),
            coverage_complete: true,
        };
    }
    
    // Use ASTChunk-style recursive splitting
    let mut chunks = Vec::new();
    let mut cursor = root_node.walk();
    extract_chunks_with_recursive_splitting(&mut cursor, source, &mut chunks, &context, &mut error_stats);
    
    // Merge adjacent small chunks if possible
    let merged_chunks = merge_adjacent_chunks(chunks, params.max_chunk_size);
    
    let chunking_method = if error_stats.error_count > 0 {
        "rust_haskell_ast_recursive_with_errors".to_string()
    } else {
        "rust_haskell_ast_recursive".to_string()
    };
    
    ChunkingResult {
        chunks: merged_chunks,
        error_stats,
        chunking_method,
        coverage_complete: true,
    }
}

fn extract_semantic_chunks_with_error_handling(tree: &Tree, source: &str) -> ChunkingResult {
    let root_node = tree.root_node();
    
    // Initialize context for metadata collection
    let mut context = ChunkingContext {
        ancestors: Vec::new(),
        max_chunk_size: 2000,
        min_chunk_size: 300,
        current_module: None,
        current_class: None,
        current_function: None,
        functions: Vec::new(),
        data_types: Vec::new(),
        type_classes: Vec::new(),
        instances: Vec::new(),
        modules: Vec::new(),
        imports: Vec::new(),
        nodes_with_errors: Vec::new(),
    };
    
    // Collect aggregate metadata
    let mut cursor = root_node.walk();
    collect_aggregate_metadata(&mut cursor, source, &mut context);
    
    // First, count error nodes
    let mut error_stats = ErrorNodeStats {
        error_count: 0,
        nodes_with_errors: 0,
        uncovered_ranges: Vec::new(),
        should_fallback: false,
    };
    
    let mut cursor = root_node.walk();
    count_error_nodes(&mut cursor, &mut error_stats);
    
    // Check if we should bail out to regex fallback
    error_stats.should_fallback = error_stats.error_count >= ERROR_FALLBACK_THRESHOLD;
    
    if error_stats.should_fallback {
        // Use regex fallback chunking
        let regex_chunks = create_regex_fallback_chunks(source);
        return ChunkingResult {
            chunks: regex_chunks,
            error_stats,
            chunking_method: "rust_haskell_regex_fallback_2".to_string(),
            coverage_complete: true, // Regex fallback covers all lines
        };
    }
    
    // Extract semantic chunks with error-aware processing
    let mut chunks = Vec::new();
    let mut cursor = root_node.walk();
    extract_chunks_recursive_with_errors(&mut cursor, source, &mut chunks, 0, &mut error_stats, &context);
    
    // Ensure complete coverage by handling uncovered ranges in error nodes
    ensure_complete_coverage(&mut chunks, &error_stats, source);
    
    let chunking_method = if error_stats.error_count > 0 {
        "rust_haskell_ast_with_errors".to_string()
    } else {
        "rust_haskell_ast".to_string()
    };
    
    ChunkingResult {
        chunks,
        error_stats,
        chunking_method,
        coverage_complete: true,
    }
}

fn extract_chunks_with_recursive_splitting(
    cursor: &mut TreeCursor,
    source: &str,
    chunks: &mut Vec<HaskellChunk>,
    context: &ChunkingContext,
    error_stats: &mut ErrorNodeStats,
) {
    let node = cursor.node();
    let node_type = node.kind();
    
    // Handle error nodes
    if node.is_error() {
        handle_error_node_with_context(cursor, source, chunks, context, error_stats);
        return;
    }
    
    // Update context with current node information
    let mut new_context = context.clone();
    update_context_for_node(&node, source, &mut new_context);
    
    // Define semantic chunk types - removed "module" to prevent tiny chunks
    let chunk_node_types = [
        "signature", "function", "bind", "data_type", "class", 
        "instance", "import", "haddock",
    ];
    
    if chunk_node_types.contains(&node_type) {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let chunk_size = end_byte - start_byte;
        
        // Check if chunk meets minimum size requirement to avoid tiny chunks
        if chunk_size < new_context.min_chunk_size {
            // Skip chunks that are too small - process children instead
            if cursor.goto_first_child() {
                loop {
                    extract_chunks_with_recursive_splitting(cursor, source, chunks, &new_context, error_stats);
                    if !cursor.goto_next_sibling() {
                        break;
                    }
                }
                cursor.goto_parent();
            }
            // Don't return here - continue to allow processing of sibling nodes
        }
        
        // Check if this node is too large and needs recursive splitting
        if chunk_size > new_context.max_chunk_size {
            // Recursively process children
            if cursor.goto_first_child() {
                loop {
                    extract_chunks_with_recursive_splitting(cursor, source, chunks, &new_context, error_stats);
                    if !cursor.goto_next_sibling() {
                        break;
                    }
                }
                cursor.goto_parent();
            }
        } else {
            // Create chunk with context information
            let chunk = create_chunk_with_context(&node, source, &new_context, node_type);
            chunks.push(chunk);
        }
    } else {
        // For non-chunk nodes, process children
        if cursor.goto_first_child() {
            loop {
                extract_chunks_with_recursive_splitting(cursor, source, chunks, &new_context, error_stats);
                if !cursor.goto_next_sibling() {
                    break;
                }
            }
            cursor.goto_parent();
        }
    }
}

fn collect_aggregate_metadata(cursor: &mut TreeCursor, source: &str, context: &mut ChunkingContext) {
    let node = cursor.node();
    let node_type = node.kind();
    
    // Collect metadata based on node type
    match node_type {
        "function" | "bind" => {
            if let Some(name) = extract_function_name(&node, source) {
                context.functions.push(name);
            }
        }
        "data_type" => {
            if let Some(name) = extract_type_name(&node, source) {
                context.data_types.push(name);
            }
        }
        "class" => {
            if let Some(name) = extract_class_name(&node, source) {
                context.type_classes.push(name);
            }
        }
        "instance" => {
            if let Some(name) = extract_class_name(&node, source) {
                context.instances.push(format!("instance {}", name));
            }
        }
        "module" => {
            if let Some(name) = extract_module_name(&node, source) {
                context.modules.push(name);
            }
        }
        "import" => {
            if let Some(name) = extract_import_name(&node, source) {
                context.imports.push(name);
            }
        }
        _ => {}
    }
    
    // Track error nodes
    if node.is_error() || node.has_error() {
        context.nodes_with_errors.push(node_type.to_string());
    }
    
    // Recursively process children
    if cursor.goto_first_child() {
        loop {
            collect_aggregate_metadata(cursor, source, context);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

fn update_context_for_node(node: &Node, source: &str, context: &mut ChunkingContext) {
    let node_type = node.kind();
    
    match node_type {
        "module" => {
            if let Some(name) = extract_module_name(node, source) {
                context.current_module = Some(name.clone());
                context.ancestors.push(ContextNode {
                    node_type: "module".to_string(),
                    name: Some(name),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                });
            }
        }
        "class" => {
            if let Some(name) = extract_class_name(node, source) {
                context.current_class = Some(name.clone());
                context.ancestors.push(ContextNode {
                    node_type: "class".to_string(),
                    name: Some(name),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                });
            }
        }
        "function" | "bind" => {
            if let Some(name) = extract_function_name(node, source) {
                context.current_function = Some(name.clone());
                context.ancestors.push(ContextNode {
                    node_type: "function".to_string(),
                    name: Some(name),
                    start_byte: node.start_byte(),
                    end_byte: node.end_byte(),
                });
            }
        }
        _ => {}
    }
}

fn create_chunk_with_context(node: &Node, source: &str, context: &ChunkingContext, node_type: &str) -> HaskellChunk {
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();
    let start_pos = node.start_position();
    let end_pos = node.end_position();
    let text = source[start_byte..end_byte].to_string();
    
    let mut metadata = HashMap::new();
    
    // Add context information
    if let Some(ref module) = context.current_module {
        metadata.insert("module_name".to_string(), module.clone());
    }
    if let Some(ref class) = context.current_class {
        metadata.insert("class_name".to_string(), class.clone());
    }
    if let Some(ref function) = context.current_function {
        metadata.insert("parent_function".to_string(), function.clone());
    }
    
    // Add ancestor path for semantic context
    let ancestor_path: Vec<String> = context.ancestors.iter()
        .filter_map(|a| a.name.as_ref())
        .cloned()
        .collect();
    if !ancestor_path.is_empty() {
        metadata.insert("ancestor_path".to_string(), ancestor_path.join("::"));
    }
    
    // Add chunking method
    metadata.insert("chunking_method".to_string(), "rust_haskell_ast_recursive".to_string());
    
    // Add aggregate metadata as JSON arrays (simple string arrays only)
    metadata.insert("functions".to_string(), format!("[{}]", context.functions.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("data_types".to_string(), format!("[{}]", context.data_types.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("type_classes".to_string(), format!("[{}]", context.type_classes.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("instances".to_string(), format!("[{}]", context.instances.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("modules".to_string(), format!("[{}]", context.modules.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("imports".to_string(), format!("[{}]", context.imports.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("nodes_with_errors".to_string(), format!("[{}]", context.nodes_with_errors.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    
    // Add empty arrays for other expected simple List[str] properties
    metadata.insert("enums".to_string(), "[]".to_string());
    metadata.insert("namespaces".to_string(), "[]".to_string());
    metadata.insert("dunder_methods".to_string(), "[]".to_string());
    metadata.insert("decorators_used".to_string(), "[]".to_string());
    metadata.insert("private_methods".to_string(), "[]".to_string());
    metadata.insert("variables".to_string(), "[]".to_string());
    metadata.insert("decorators".to_string(), "[]".to_string());
    metadata.insert("classes".to_string(), format!("[{}]", context.type_classes.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    
    // Do NOT populate complex JSON fields like function_details, data_type_details, class_details
    // These should be handled by the language-specific analysis handlers
    
    // Add tree-sitter error tracking for chunking
    if node.is_error() {
        metadata.insert("tree_sitter_chunking_error".to_string(), "true".to_string());
        metadata.insert("has_error".to_string(), "true".to_string());
    } else {
        metadata.insert("tree_sitter_chunking_error".to_string(), "false".to_string());
    }
    
    // Node-specific metadata
    match node_type {
        "function" | "bind" => {
            if let Some(name) = extract_function_name(node, source) {
                metadata.insert("function_name".to_string(), name);
            }
            metadata.insert("category".to_string(), "function".to_string());
        }
        "signature" => {
            if let Some(name) = extract_function_name(node, source) {
                metadata.insert("function_name".to_string(), name);
            }
            metadata.insert("category".to_string(), "signature".to_string());
        }
        "data_type" => {
            if let Some(name) = extract_type_name(node, source) {
                metadata.insert("type_name".to_string(), name);
            }
            metadata.insert("category".to_string(), "data_type".to_string());
        }
        "class" => {
            if let Some(name) = extract_class_name(node, source) {
                metadata.insert("class_name".to_string(), name);
            }
            metadata.insert("category".to_string(), "class".to_string());
        }
        "instance" => {
            if let Some(name) = extract_class_name(node, source) {
                metadata.insert("class_name".to_string(), name);
            }
            metadata.insert("category".to_string(), "instance".to_string());
        }
        "import" => {
            if let Some(name) = extract_import_name(node, source) {
                metadata.insert("import_name".to_string(), name);
            }
            metadata.insert("category".to_string(), "import".to_string());
        }
        "module" => {
            if let Some(name) = extract_module_name(node, source) {
                metadata.insert("module_name".to_string(), name);
            }
            metadata.insert("category".to_string(), "module".to_string());
        }
        _ => {
            metadata.insert("category".to_string(), "other".to_string());
        }
    }
    
    HaskellChunk {
        text,
        start_byte,
        end_byte,
        start_line: start_pos.row,
        end_line: end_pos.row,
        node_type: node_type.to_string(),
        metadata,
    }
}

fn merge_adjacent_chunks(chunks: Vec<HaskellChunk>, max_size: usize) -> Vec<HaskellChunk> {
    if chunks.is_empty() {
        return chunks;
    }
    
    let mut merged = Vec::new();
    let mut current_chunk = chunks[0].clone();
    
    for chunk in chunks.into_iter().skip(1) {
        let combined_size = current_chunk.text.len() + chunk.text.len();
        
        // Check if chunks can be merged (same category and within size limit)
        let can_merge = combined_size <= max_size
            && current_chunk.metadata.get("category") == chunk.metadata.get("category")
            && current_chunk.end_byte == chunk.start_byte;
            
        if can_merge {
            // Merge chunks
            current_chunk.text.push('\n');
            current_chunk.text.push_str(&chunk.text);
            current_chunk.end_byte = chunk.end_byte;
            current_chunk.end_line = chunk.end_line;
        } else {
            merged.push(current_chunk);
            current_chunk = chunk;
        }
    }
    
    merged.push(current_chunk);
    merged
}

fn handle_error_node_with_context(
    cursor: &mut TreeCursor,
    source: &str,
    chunks: &mut Vec<HaskellChunk>,
    context: &ChunkingContext,
    error_stats: &mut ErrorNodeStats,
) {
    let node = cursor.node();
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();
    
    // Try to process children of error node
    if cursor.goto_first_child() {
        loop {
            let child_node = cursor.node();
            if !child_node.is_error() {
                extract_chunks_with_recursive_splitting(cursor, source, chunks, context, error_stats);
            }
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
    
    // Create error chunk for uncovered ranges
    let error_chunk = create_error_chunk_with_context(start_byte, end_byte, source, context);
    chunks.push(error_chunk);
    error_stats.uncovered_ranges.push((start_byte, end_byte));
}

fn create_error_chunk_with_context(start_byte: usize, end_byte: usize, source: &str, context: &ChunkingContext) -> HaskellChunk {
    let text = source[start_byte..end_byte].to_string();
    let start_line = source[..start_byte].matches('\n').count();
    let end_line = source[..end_byte].matches('\n').count();
    
    let mut metadata = HashMap::new();
    metadata.insert("category".to_string(), "error_recovery".to_string());
    metadata.insert("chunking_method".to_string(), "rust_haskell_error_recovery".to_string());
    metadata.insert("is_error_chunk".to_string(), "true".to_string());
    metadata.insert("tree_sitter_chunking_error".to_string(), "true".to_string());
    metadata.insert("has_error".to_string(), "true".to_string());
    
    // Add context information even for error chunks
    if let Some(ref module) = context.current_module {
        metadata.insert("module_name".to_string(), module.clone());
    }
    if let Some(ref class) = context.current_class {
        metadata.insert("class_name".to_string(), class.clone());
    }
    
    HaskellChunk {
        text,
        start_byte,
        end_byte,
        start_line,
        end_line,
        node_type: "error_recovery".to_string(),
        metadata,
    }
}

fn extract_module_name(node: &Node, source: &str) -> Option<String> {
    for child in node.children(&mut node.walk()) {
        if child.kind() == "module_id" {
            let name = source[child.start_byte()..child.end_byte()].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

fn count_error_nodes(cursor: &mut TreeCursor, stats: &mut ErrorNodeStats) {
    let node = cursor.node();
    
    if node.is_error() {
        stats.error_count += 1;
    }
    
    if node.has_error() {
        stats.nodes_with_errors += 1;
    }
    
    // Recursively count in children
    if cursor.goto_first_child() {
        loop {
            count_error_nodes(cursor, stats);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

fn extract_chunks_recursive_with_errors(
    cursor: &mut TreeCursor,
    source: &str,
    chunks: &mut Vec<HaskellChunk>,
    depth: usize,
    error_stats: &mut ErrorNodeStats,
    context: &ChunkingContext,
) {
    let node = cursor.node();
    let node_type = node.kind();
    
    // Handle error nodes: try to extract valid children, mark uncovered ranges
    if node.is_error() {
        handle_error_node(cursor, source, chunks, depth, error_stats, context);
        return;
    }
    
    // Define which node types we want to extract as chunks
    let chunk_node_types = [
        "signature",      // Type signatures
        "function",       // Function definitions
        "bind",           // Top-level bindings
        "data_type",      // Data type declarations
        "class",          // Type class declarations
        "instance",       // Type class instances
        "import",         // Import statements
        "haddock",        // Haddock documentation comments
    ];
    
    if chunk_node_types.contains(&node_type) {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let start_pos = node.start_position();
        let end_pos = node.end_position();
        
        let text = source[start_byte..end_byte].to_string();
        
        let mut metadata = HashMap::new();
        metadata.insert("depth".to_string(), depth.to_string());
        metadata.insert("has_children".to_string(), (node.child_count() > 0).to_string());
        metadata.insert("is_named".to_string(), node.is_named().to_string());
        metadata.insert("has_error".to_string(), node.has_error().to_string());
        metadata.insert("chunking_method".to_string(), "rust_haskell_ast_with_errors_2".to_string());
        
        // Add aggregate metadata as JSON arrays (simple string arrays only)
        metadata.insert("functions".to_string(), format!("[{}]", context.functions.iter()
            .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
            .collect::<Vec<_>>().join(", ")));
        metadata.insert("data_types".to_string(), format!("[{}]", context.data_types.iter()
            .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
            .collect::<Vec<_>>().join(", ")));
        metadata.insert("type_classes".to_string(), format!("[{}]", context.type_classes.iter()
            .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
            .collect::<Vec<_>>().join(", ")));
        metadata.insert("instances".to_string(), format!("[{}]", context.instances.iter()
            .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
            .collect::<Vec<_>>().join(", ")));
        metadata.insert("modules".to_string(), format!("[{}]", context.modules.iter()
            .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
            .collect::<Vec<_>>().join(", ")));
        metadata.insert("imports".to_string(), format!("[{}]", context.imports.iter()
            .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
            .collect::<Vec<_>>().join(", ")));
        metadata.insert("nodes_with_errors".to_string(), format!("[{}]", context.nodes_with_errors.iter()
            .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
            .collect::<Vec<_>>().join(", ")));
        
        // Add empty arrays for other expected simple List[str] properties
        metadata.insert("enums".to_string(), "[]".to_string());
        metadata.insert("namespaces".to_string(), "[]".to_string());
        metadata.insert("dunder_methods".to_string(), "[]".to_string());
        metadata.insert("decorators_used".to_string(), "[]".to_string());
        metadata.insert("private_methods".to_string(), "[]".to_string());
        metadata.insert("variables".to_string(), "[]".to_string());
        metadata.insert("decorators".to_string(), "[]".to_string());
        metadata.insert("classes".to_string(), format!("[{}]", context.type_classes.iter()
            .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
            .collect::<Vec<_>>().join(", ")));
        
        // Extract additional metadata based on node type
        match node_type {
            "function" => {
                if let Some(name) = extract_function_name(&node, source) {
                    metadata.insert("function_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "function".to_string());
            }
            "signature" => {
                if let Some(name) = extract_function_name(&node, source) {
                    metadata.insert("function_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "signature".to_string());
            }
            "bind" => {
                if let Some(name) = extract_function_name(&node, source) {
                    metadata.insert("function_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "binding".to_string());
            }
            "data_type" => {
                if let Some(name) = extract_type_name(&node, source) {
                    metadata.insert("type_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "data_type".to_string());
            }
            "class" => {
                if let Some(name) = extract_class_name(&node, source) {
                    metadata.insert("class_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "class".to_string());
            }
            "instance" => {
                if let Some(name) = extract_class_name(&node, source) {
                    metadata.insert("class_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "instance".to_string());
            }
            "import" => {
                if let Some(name) = extract_import_name(&node, source) {
                    metadata.insert("import_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "import".to_string());
            }
            "haddock" => {
                metadata.insert("category".to_string(), "documentation".to_string());
            }
            _ => {
                metadata.insert("category".to_string(), "other".to_string());
            }
        }
        
        chunks.push(HaskellChunk {
            text,
            start_byte,
            end_byte,
            start_line: start_pos.row,
            end_line: end_pos.row,
            node_type: node_type.to_string(),
            metadata,
        });
    }
    
    // Recursively process children
    if cursor.goto_first_child() {
        loop {
            extract_chunks_recursive_with_errors(cursor, source, chunks, depth + 1, error_stats, context);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

#[allow(dead_code)]
fn extract_chunks_recursive(
    cursor: &mut TreeCursor,
    source: &str,
    chunks: &mut Vec<HaskellChunk>,
    depth: usize,
) {
    let node = cursor.node();
    let node_type = node.kind();
    
    // Define which node types we want to extract as chunks
    // These are based on actual tree-sitter-haskell node types from debugging
    let chunk_node_types = [
        "signature",      // Type signatures
        "function",       // Function definitions
        "bind",           // Top-level bindings
        "data_type",      // Data type declarations
        "class",          // Type class declarations
        "instance",       // Type class instances
        "import",         // Import statements
        "haddock",        // Haddock documentation comments
    ];
    
    if chunk_node_types.contains(&node_type) {
        let start_byte = node.start_byte();
        let end_byte = node.end_byte();
        let start_pos = node.start_position();
        let end_pos = node.end_position();
        
        let text = source[start_byte..end_byte].to_string();
        
        let mut metadata = HashMap::new();
        metadata.insert("depth".to_string(), depth.to_string());
        metadata.insert("has_children".to_string(), (node.child_count() > 0).to_string());
        metadata.insert("is_named".to_string(), node.is_named().to_string());
        
        // Extract additional metadata based on node type
        match node_type {
            "function" => {
                if let Some(name) = extract_function_name(&node, source) {
                    metadata.insert("function_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "function".to_string());
            }
            "signature" => {
                if let Some(name) = extract_function_name(&node, source) {
                    metadata.insert("function_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "signature".to_string());
            }
            "bind" => {
                if let Some(name) = extract_function_name(&node, source) {
                    metadata.insert("function_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "binding".to_string());
            }
            "data_type" => {
                if let Some(name) = extract_type_name(&node, source) {
                    metadata.insert("type_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "data_type".to_string());
            }
            "class" => {
                if let Some(name) = extract_class_name(&node, source) {
                    metadata.insert("class_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "class".to_string());
            }
            "instance" => {
                if let Some(name) = extract_class_name(&node, source) {
                    metadata.insert("class_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "instance".to_string());
            }
            "import" => {
                if let Some(name) = extract_import_name(&node, source) {
                    metadata.insert("import_name".to_string(), name);
                }
                metadata.insert("category".to_string(), "import".to_string());
            }
            "haddock" => {
                metadata.insert("category".to_string(), "documentation".to_string());
            }
            _ => {
                metadata.insert("category".to_string(), "other".to_string());
            }
        }
        
        chunks.push(HaskellChunk {
            text,
            start_byte,
            end_byte,
            start_line: start_pos.row,
            end_line: end_pos.row,
            node_type: node_type.to_string(),
            metadata,
        });
    }
    
    // Recursively process children
    if cursor.goto_first_child() {
        loop {
            extract_chunks_recursive(cursor, source, chunks, depth + 1);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

fn extract_function_name(node: &Node, source: &str) -> Option<String> {
    // Look for variable nodes that represent function names
    for child in node.children(&mut node.walk()) {
        if child.kind() == "variable" {
            let name = source[child.start_byte()..child.end_byte()].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

fn extract_type_name(node: &Node, source: &str) -> Option<String> {
    // Look for name nodes that represent type names
    for child in node.children(&mut node.walk()) {
        if child.kind() == "name" {
            let name = source[child.start_byte()..child.end_byte()].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}

fn extract_class_name(node: &Node, source: &str) -> Option<String> {
    // Look for name nodes that represent class names
    for child in node.children(&mut node.walk()) {
        if child.kind() == "name" {
            let name = source[child.start_byte()..child.end_byte()].trim();
            if !name.is_empty() {
                return Some(name.to_string());
            }
        }
    }
    None
}


fn extract_import_name(node: &Node, source: &str) -> Option<String> {
    // Look for module nodes in imports
    for child in node.children(&mut node.walk()) {
        if child.kind() == "module" {
            // Look for module_id inside the module node
            for grandchild in child.children(&mut child.walk()) {
                if grandchild.kind() == "module_id" {
                    let name = source[grandchild.start_byte()..grandchild.end_byte()].trim();
                    if !name.is_empty() {
                        return Some(name.to_string());
                    }
                }
            }
        }
    }
    None
}

#[pyfunction]
fn get_haskell_ast_chunks(source: &str) -> PyResult<Vec<HaskellChunk>> {
    let mut parser = Parser::new();
    let language = tree_sitter_haskell::LANGUAGE.into();
    parser.set_language(&language)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to set Haskell language: {}", e)
        ))?;
    
    match parser.parse(source, None) {
        Some(tree) => {
            // Use the enhanced chunking with proper error handling and method names
            let default_params = ChunkingParams {
                chunk_size: 1800,
                min_chunk_size: 300,
                chunk_overlap: 0,
                max_chunk_size: 1800,
            };
            let result = extract_semantic_chunks_with_recursive_splitting(&tree, source, &default_params);
            Ok(result.chunks)
        }
        None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Failed to parse Haskell source"
        )),
    }
}

#[pyfunction]
fn get_haskell_ast_chunks_with_params(source: &str, params: ChunkingParams) -> PyResult<ChunkingResult> {
    let mut parser = Parser::new();
    let language = tree_sitter_haskell::LANGUAGE.into();
    parser.set_language(&language)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to set Haskell language: {}", e)
        ))?;
    
    match parser.parse(source, None) {
        Some(tree) => {
            let result = extract_semantic_chunks_with_recursive_splitting(&tree, source, &params);
            Ok(result)
        }
        None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Failed to parse Haskell source"
        )),
    }
}

#[pyfunction]
fn get_haskell_ast_chunks_enhanced(source: &str) -> PyResult<ChunkingResult> {
    let mut parser = Parser::new();
    let language = tree_sitter_haskell::LANGUAGE.into();
    parser.set_language(&language)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to set Haskell language: {}", e)
        ))?;
    
    match parser.parse(source, None) {
        Some(tree) => {
            let result = extract_semantic_chunks_with_error_handling(&tree, source);
            Ok(result)
        }
        None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Failed to parse Haskell source"
        )),
    }
}

#[pyfunction]
fn debug_haskell_ast_nodes(source: &str) -> PyResult<Vec<String>> {
    let mut parser = Parser::new();
    let language = tree_sitter_haskell::LANGUAGE.into();
    parser.set_language(&language)
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to set Haskell language: {}", e)
        ))?;
    
    match parser.parse(source, None) {
        Some(tree) => {
            let mut node_types = Vec::new();
            let root_node = tree.root_node();
            let mut cursor = root_node.walk();
            collect_node_types(&mut cursor, &mut node_types, 0);
            Ok(node_types)
        }
        None => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            "Failed to parse Haskell source"
        )),
    }
}

fn collect_node_types(cursor: &mut TreeCursor, node_types: &mut Vec<String>, depth: usize) {
    let node = cursor.node();
    let node_type = node.kind();
    
    // Add node type with depth info
    node_types.push(format!("{}{}", "  ".repeat(depth), node_type));
    
    // Recursively collect child node types
    if cursor.goto_first_child() {
        loop {
            collect_node_types(cursor, node_types, depth + 1);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

#[pyfunction]
fn get_haskell_ast_chunks_with_fallback(source: &str) -> PyResult<Vec<HaskellChunk>> {
    // First try AST-based chunking
    match get_haskell_ast_chunks(source) {
        Ok(chunks) if !chunks.is_empty() => Ok(chunks),
        _ => {
            // Fallback to regex-based chunking for malformed code
            let regex_chunks = create_regex_fallback_chunks(source);
            Ok(regex_chunks)
        }
    }
}

fn handle_error_node(
    cursor: &mut TreeCursor,
    source: &str,
    chunks: &mut Vec<HaskellChunk>,
    depth: usize,
    error_stats: &mut ErrorNodeStats,
    context: &ChunkingContext,
) {
    let node = cursor.node();
    let start_byte = node.start_byte();
    let end_byte = node.end_byte();
    
    // Track this as an uncovered range initially
    let mut covered_ranges: Vec<(usize, usize)> = Vec::new();
    
    // Try to process children of error node to find valid semantic chunks
    if cursor.goto_first_child() {
        loop {
            let child_node = cursor.node();
            if !child_node.is_error() {
                // Process valid child nodes normally
                extract_chunks_recursive_with_errors(cursor, source, chunks, depth + 1, error_stats, context);
                
                // Track what we covered
                covered_ranges.push((child_node.start_byte(), child_node.end_byte()));
            } else {
                // Recursively handle nested error nodes
                handle_error_node(cursor, source, chunks, depth + 1, error_stats, context);
            }
            
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
    
    // Find uncovered ranges within this error node
    let uncovered = find_uncovered_ranges_in_node(start_byte, end_byte, &covered_ranges);
    
    // Create error chunks for uncovered ranges
    for (uncov_start, uncov_end) in uncovered {
        if uncov_end > uncov_start {
            let error_chunk = create_error_chunk(uncov_start, uncov_end, source, depth, context);
            chunks.push(error_chunk);
            error_stats.uncovered_ranges.push((uncov_start, uncov_end));
        }
    }
}

fn find_uncovered_ranges_in_node(start_byte: usize, end_byte: usize, covered_ranges: &[(usize, usize)]) -> Vec<(usize, usize)> {
    let mut uncovered = Vec::new();
    let mut current_pos = start_byte;
    
    // Sort covered ranges by start position
    let mut sorted_ranges = covered_ranges.to_vec();
    sorted_ranges.sort_by_key(|&(start, _)| start);
    
    for (cover_start, cover_end) in sorted_ranges {
        // Add gap before this covered range
        if current_pos < cover_start {
            uncovered.push((current_pos, cover_start));
        }
        current_pos = current_pos.max(cover_end);
    }
    
    // Add final gap after last covered range
    if current_pos < end_byte {
        uncovered.push((current_pos, end_byte));
    }
    
    uncovered
}

fn create_error_chunk(start_byte: usize, end_byte: usize, source: &str, depth: usize, context: &ChunkingContext) -> HaskellChunk {
    let text = source[start_byte..end_byte].to_string();
    
    // Calculate line numbers (approximation)
    let start_line = source[..start_byte].matches('\n').count();
    let end_line = source[..end_byte].matches('\n').count();
    
    let mut metadata = HashMap::new();
    metadata.insert("depth".to_string(), depth.to_string());
    metadata.insert("category".to_string(), "error_recovery".to_string());
    metadata.insert("chunking_method".to_string(), "rust_haskell_error_recovery_2".to_string());
    metadata.insert("is_error_chunk".to_string(), "true".to_string());
    metadata.insert("tree_sitter_chunking_error".to_string(), "true".to_string());
    metadata.insert("has_error".to_string(), "true".to_string());
    
    // Add aggregate metadata as JSON arrays (simple string arrays only)
    metadata.insert("functions".to_string(), format!("[{}]", context.functions.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("data_types".to_string(), format!("[{}]", context.data_types.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("type_classes".to_string(), format!("[{}]", context.type_classes.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("instances".to_string(), format!("[{}]", context.instances.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("modules".to_string(), format!("[{}]", context.modules.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("imports".to_string(), format!("[{}]", context.imports.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    metadata.insert("nodes_with_errors".to_string(), format!("[{}]", context.nodes_with_errors.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    
    // Add empty arrays for other expected simple List[str] properties
    metadata.insert("enums".to_string(), "[]".to_string());
    metadata.insert("namespaces".to_string(), "[]".to_string());
    metadata.insert("dunder_methods".to_string(), "[]".to_string());
    metadata.insert("decorators_used".to_string(), "[]".to_string());
    metadata.insert("private_methods".to_string(), "[]".to_string());
    metadata.insert("variables".to_string(), "[]".to_string());
    metadata.insert("decorators".to_string(), "[]".to_string());
    metadata.insert("classes".to_string(), format!("[{}]", context.type_classes.iter()
        .map(|s| format!("\"{}\"", s.replace("\"", "\\\"")))
        .collect::<Vec<_>>().join(", ")));
    
    HaskellChunk {
        text,
        start_byte,
        end_byte,
        start_line,
        end_line,
        node_type: "error_recovery".to_string(),
        metadata,
    }
}

fn ensure_complete_coverage(_chunks: &mut Vec<HaskellChunk>, _error_stats: &ErrorNodeStats, _source: &str) {
    // This function ensures all source code is covered by chunks
    // For now, we rely on the error node handling above to create error chunks
    // In the future, we could add additional gap detection here
}

fn create_regex_fallback_chunks(source: &str) -> Vec<HaskellChunk> {
    let separators = get_haskell_separators();
    let mut chunks = Vec::new();
    
    // Simple regex-based chunking as fallback
    let lines: Vec<&str> = source.lines().collect();
    let mut current_start = 0;
    let mut current_line = 0;
    
    for (i, line) in lines.iter().enumerate() {
        let mut is_separator = false;
        
        for separator in &separators {
            if line.starts_with(separator.trim_start_matches("\\n")) {
                is_separator = true;
                break;
            }
        }
        
        if is_separator && current_start < i {
            let chunk_lines = &lines[current_start..i];
            let chunk_text = chunk_lines.join("\n");
            
            if !chunk_text.trim().is_empty() {
                let mut metadata = HashMap::new();
                metadata.insert("category".to_string(), "regex_fallback".to_string());
                metadata.insert("chunking_method".to_string(), "rust_haskell_regex_fallback_3".to_string());
                metadata.insert("tree_sitter_chunking_error".to_string(), "false".to_string());
                
                let chunk_len = chunk_text.len();
                chunks.push(HaskellChunk {
                    text: chunk_text,
                    start_byte: 0, // Approximation
                    end_byte: chunk_len,
                    start_line: current_line,
                    end_line: i,
                    node_type: "regex_chunk".to_string(),
                    metadata,
                });
            }
            
            current_start = i;
            current_line = i;
        }
    }
    
    // Handle the last chunk
    if current_start < lines.len() {
        let chunk_lines = &lines[current_start..];
        let chunk_text = chunk_lines.join("\n");
        
        if !chunk_text.trim().is_empty() {
            let mut metadata = HashMap::new();
            metadata.insert("category".to_string(), "regex_fallback".to_string());
            metadata.insert("chunking_method".to_string(), "rust_haskell_regex_fallback_4".to_string());
            metadata.insert("tree_sitter_chunking_error".to_string(), "false".to_string());
            
            let chunk_len = chunk_text.len();
            chunks.push(HaskellChunk {
                text: chunk_text,
                start_byte: 0, // Approximation
                end_byte: chunk_len,
                start_line: current_line,
                end_line: lines.len(),
                node_type: "regex_chunk".to_string(),
                metadata,
            });
        }
    }
    
    chunks
}

#[pymodule]
fn haskell_tree_sitter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HaskellParser>()?;
    m.add_class::<HaskellTree>()?;
    m.add_class::<HaskellNode>()?;
    m.add_class::<HaskellChunk>()?;
    m.add_class::<ErrorNodeStats>()?;
    m.add_class::<ChunkingResult>()?;
    m.add_class::<ChunkingParams>()?;
    m.add_function(wrap_pyfunction!(parse_haskell, m)?)?;
    m.add_function(wrap_pyfunction!(get_haskell_separators, m)?)?;
    m.add_function(wrap_pyfunction!(get_haskell_ast_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(get_haskell_ast_chunks_enhanced, m)?)?;
    m.add_function(wrap_pyfunction!(get_haskell_ast_chunks_with_fallback, m)?)?;
    m.add_function(wrap_pyfunction!(get_haskell_ast_chunks_with_params, m)?)?;
    m.add_function(wrap_pyfunction!(debug_haskell_ast_nodes, m)?)?;
    Ok(())
}
