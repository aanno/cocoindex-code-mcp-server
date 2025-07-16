use pyo3::prelude::*;
use pyo3::types::PyModule;
use tree_sitter::{Parser, Tree};

#[pyclass]
pub struct HaskellParser {
    parser: Parser,
}

#[pymethods]
impl HaskellParser {
    #[new]
    fn new() -> PyResult<Self> {
        let mut parser = Parser::new();
        let language = tree_sitter_haskell::language();
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

#[pymodule]
fn haskell_tree_sitter(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<HaskellParser>()?;
    m.add_class::<HaskellTree>()?;
    m.add_class::<HaskellNode>()?;
    m.add_function(wrap_pyfunction!(parse_haskell, m)?)?;
    m.add_function(wrap_pyfunction!(get_haskell_separators, m)?)?;
    Ok(())
}