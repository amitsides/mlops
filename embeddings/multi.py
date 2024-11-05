text_config = TextEmbeddingConfig(model_name="sentence-transformers/all-mpnet-base-v2", max_length=256)
graph_config = GraphEmbeddingConfig(max_nodes=500, graph_conv_layers=[128, 64])
molecule_config = MoleculeEmbeddingConfig(featurizer="ECFP", radius=3)

multi_modal_config = MultiModalEmbeddingConfig(
    text_config=text_config,
    graph_config=graph_config,
    molecule_config=molecule_config,
    fusion_method="attention",
    output_dim=768
)

pipeline = EmbeddingPipeline(config=multi_modal_config)