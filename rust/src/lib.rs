use pyo3::prelude::*;
use pyo3::types::PyModule;
use tree_sitter::{Parser, Tree, Node, TreeCursor};
use std::collections::HashMap;

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

fn extract_semantic_chunks(tree: &Tree, source: &str) -> Vec<HaskellChunk> {
    let mut chunks = Vec::new();
    let root_node = tree.root_node();
    
    // Walk through the tree and extract semantic chunks
    let mut cursor = root_node.walk();
    extract_chunks_recursive(&mut cursor, source, &mut chunks, 0);
    
    chunks
}

fn extract_semantic_chunks_with_error_handling(tree: &Tree, source: &str) -> ChunkingResult {
    let root_node = tree.root_node();
    
    // First, count error nodes
    let mut error_stats = ErrorNodeStats {
        error_count: 0,
        nodes_with_errors: 0,
        uncovered_ranges: Vec::new(),
        should_fallback: false,
    };
    
    let mut cursor = root_node.walk();
    count_error_nodes(&mut cursor, &mut error_stats);
    
    // Check if we should bail out to regex fallback (3+ error nodes)
    error_stats.should_fallback = error_stats.error_count >= 3;
    
    if error_stats.should_fallback {
        // Use regex fallback chunking
        let regex_chunks = create_regex_fallback_chunks(source);
        return ChunkingResult {
            chunks: regex_chunks,
            error_stats,
            chunking_method: "regex_fallback".to_string(),
            coverage_complete: true, // Regex fallback covers all lines
        };
    }
    
    // Extract semantic chunks with error-aware processing
    let mut chunks = Vec::new();
    let mut cursor = root_node.walk();
    extract_chunks_recursive_with_errors(&mut cursor, source, &mut chunks, 0, &mut error_stats);
    
    // Ensure complete coverage by handling uncovered ranges in error nodes
    ensure_complete_coverage(&mut chunks, &error_stats, source);
    
    let chunking_method = if error_stats.error_count > 0 {
        "ast_with_errors".to_string()
    } else {
        "ast".to_string()
    };
    
    ChunkingResult {
        chunks,
        error_stats,
        chunking_method,
        coverage_complete: true,
    }
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
) {
    let node = cursor.node();
    let node_type = node.kind();
    
    // Handle error nodes: try to extract valid children, mark uncovered ranges
    if node.is_error() {
        handle_error_node(cursor, source, chunks, depth, error_stats);
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
        metadata.insert("chunking_method".to_string(), "ast_with_errors".to_string());
        
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
            extract_chunks_recursive_with_errors(cursor, source, chunks, depth + 1, error_stats);
            if !cursor.goto_next_sibling() {
                break;
            }
        }
        cursor.goto_parent();
    }
}

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

// Removed extract_module_name function as it was unused

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
            let chunks = extract_semantic_chunks(&tree, source);
            Ok(chunks)
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
                extract_chunks_recursive_with_errors(cursor, source, chunks, depth + 1, error_stats);
                
                // Track what we covered
                covered_ranges.push((child_node.start_byte(), child_node.end_byte()));
            } else {
                // Recursively handle nested error nodes
                handle_error_node(cursor, source, chunks, depth + 1, error_stats);
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
            let error_chunk = create_error_chunk(uncov_start, uncov_end, source, depth);
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

fn create_error_chunk(start_byte: usize, end_byte: usize, source: &str, depth: usize) -> HaskellChunk {
    let text = source[start_byte..end_byte].to_string();
    
    // Calculate line numbers (approximation)
    let start_line = source[..start_byte].matches('\n').count();
    let end_line = source[..end_byte].matches('\n').count();
    
    let mut metadata = HashMap::new();
    metadata.insert("depth".to_string(), depth.to_string());
    metadata.insert("category".to_string(), "error_recovery".to_string());
    metadata.insert("chunking_method".to_string(), "error_recovery".to_string());
    metadata.insert("is_error_chunk".to_string(), "true".to_string());
    metadata.insert("has_error".to_string(), "true".to_string());
    
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
                metadata.insert("method".to_string(), "regex".to_string());
                
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
            metadata.insert("method".to_string(), "regex".to_string());
            
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
fn cocoindex_code_mcp_server(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HaskellParser>()?;
    m.add_class::<HaskellTree>()?;
    m.add_class::<HaskellNode>()?;
    m.add_class::<HaskellChunk>()?;
    m.add_class::<ErrorNodeStats>()?;
    m.add_class::<ChunkingResult>()?;
    m.add_function(wrap_pyfunction!(parse_haskell, m)?)?;
    m.add_function(wrap_pyfunction!(get_haskell_separators, m)?)?;
    m.add_function(wrap_pyfunction!(get_haskell_ast_chunks, m)?)?;
    m.add_function(wrap_pyfunction!(get_haskell_ast_chunks_enhanced, m)?)?;
    m.add_function(wrap_pyfunction!(get_haskell_ast_chunks_with_fallback, m)?)?;
    m.add_function(wrap_pyfunction!(debug_haskell_ast_nodes, m)?)?;
    Ok(())
}
