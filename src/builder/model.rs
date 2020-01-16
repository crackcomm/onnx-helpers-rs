//! Model builder.

use onnx_pb::{GraphProto, ModelProto, OperatorSetIdProto, StringStringEntryProto, Version};

const DEFAULT_OPSET_ID_VERSION: i64 = 11;

/// Model builder.
#[derive(Default, Clone)]
pub struct Model {
    graph: GraphProto,
    domain: Option<String>,
    model_version: Option<i64>,
    producer_name: Option<String>,
    producer_version: Option<String>,
    doc_string: Option<String>,
    metadata: Vec<(String, String)>,
    opset_imports: Option<Vec<OperatorSetIdProto>>,
}

impl Model {
    /// Creates a new builder.
    #[inline]
    pub fn new<G: Into<GraphProto>>(graph: G) -> Self {
        Model {
            graph: graph.into(),
            ..Model::default()
        }
    }

    /// Sets model doc_string.
    #[inline]
    pub fn domain<S: Into<String>>(mut self, domain: S) -> Self {
        self.domain = Some(domain.into());
        self
    }

    /// Sets model doc_string.
    #[inline]
    pub fn model_version(mut self, model_version: i64) -> Self {
        self.model_version = Some(model_version);
        self
    }

    /// Sets model doc_string.
    #[inline]
    pub fn producer_name<S: Into<String>>(mut self, producer_name: S) -> Self {
        self.producer_name = Some(producer_name.into());
        self
    }

    /// Sets model doc_string.
    #[inline]
    pub fn producer_version<S: Into<String>>(mut self, producer_version: S) -> Self {
        self.producer_version = Some(producer_version.into());
        self
    }

    /// Sets model doc_string.
    #[inline]
    pub fn doc_string<S: Into<String>>(mut self, doc_string: S) -> Self {
        self.doc_string = Some(doc_string.into());
        self
    }

    /// Inserts model metadata.
    #[inline]
    pub fn metadata<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        self.metadata.push((key.into(), value.into()));
        self
    }

    /// Inserts operator set import.
    #[inline]
    pub fn opset_import(mut self, opset: OperatorSetIdProto) -> Self {
        if let Some(opset_imports) = self.opset_imports.as_mut() {
            opset_imports.push(opset);
        } else {
            self.opset_imports = Some(vec![opset]);
        }
        self
    }

    /// Builds the model.
    #[inline]
    pub fn build(self) -> ModelProto {
        let opset_import = self.opset_imports.unwrap_or_else(|| {
            vec![OperatorSetIdProto {
                version: DEFAULT_OPSET_ID_VERSION,
                ..OperatorSetIdProto::default()
            }]
        });
        let metadata_props = self
            .metadata
            .into_iter()
            .map(|(k, v)| StringStringEntryProto {
                key: k.into(),
                value: v.into(),
            })
            .collect();
        ModelProto {
            ir_version: Version::IrVersion as i64,
            graph: Some(self.graph),
            domain: self.domain.unwrap_or_default(),
            doc_string: self.doc_string.unwrap_or_default(),
            producer_name: self.producer_name.unwrap_or_default(),
            producer_version: self.producer_version.unwrap_or_default(),
            model_version: self.model_version.unwrap_or_default(),
            opset_import,
            metadata_props,
            ..ModelProto::default()
        }
    }
}

impl Into<ModelProto> for Model {
    fn into(self) -> ModelProto {
        self.build()
    }
}
