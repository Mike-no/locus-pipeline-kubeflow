import os
import kfp
import yaml

# Components 
downloader_op = kfp.components.load_component_from_file(os.path.join('../kubeflow_pipeline/pandas_loading/downloader/', 'component.yaml'))
data_loading_op = kfp.components.load_component_from_file(os.path.join('./data_loading/', 'component.yaml'))
numpy_preprocessing_op = kfp.components.load_component_from_file(os.path.join('./numpy_preprocessing/', 'component.yaml'))
tensorflow_training_op = kfp.components.load_component_from_file(os.path.join('./tensorflow_training/', 'component.yaml'))
model_storing_op = kfp.components.load_component_from_file(os.path.join('../kubeflow_pipeline/model_storing/', 'component.yaml'))
kmeans_clustering_op = kfp.components.load_component_from_file(os.path.join('../kubeflow_pipeline/kmeans_clustering/', 'component.yaml'))

@kfp.dsl.pipeline(
    name = 'UC1 Functionality-1',
    description = 'Locus Pipeline for UC1 Functionality-1'
)
def enhanced_locus_pipeline(openpflow_data_url, nodes_npz_url, tokyo_osmpbf_url, tree_pkl_url, minio_url, minio_access_key, minio_secret_key):

    # Download datasets
    openpflow_trajectories = downloader_op(url = openpflow_data_url)
    nodes_npz = downloader_op(url = nodes_npz_url)
    tokyo_osmpbf = downloader_op(url = tokyo_osmpbf_url)
    tree_pkl = downloader_op(url = tree_pkl_url)

    # Get graph deployment file and 
    deployment_dict = yaml.safe_load(open('../graph/graph.yaml', 'r'))
    minio_path = deployment_dict["spec"]["predictors"][0]["graph"]["modelUri"].split("s3://")[1]
    minio_filename = deployment_dict["spec"]["predictors"][0]["componentSpecs"][0]["spec"]["containers"][0]["image"].split("/")[1].split(":")[0]

    # Load from Datasets
    data_loading = data_loading_op(openpflow_trajectories_path = openpflow_trajectories.output,
        nodes_npz_path = nodes_npz.output, tokyo_osmpbf_path = tokyo_osmpbf.output)

    # Preprocessing of the openpflow trajectories data
    numpy_preprocessing = numpy_preprocessing_op(input_path = data_loading.outputs['openpflow_trajectories_output'])

    # Training of the encoder model
    tensorflow_training = tensorflow_training_op(prep_openpflow_trajectories_path = numpy_preprocessing.output,
        nodes_npz_path = data_loading.outputs['nodes_npz_output'], tokyo_osmpbf_path = data_loading.outputs['tokyo_osmpbf_output'],
        tree_pkl_path = tree_pkl.output)

    # Save encoder model in PV
    encoder_model_storing = model_storing_op(model_path = tensorflow_training.outputs['encoder_output'], minio_url = minio_url, 
        minio_access_key = minio_access_key, minio_secret_key = minio_secret_key, minio_path = minio_path)

    # Produce clustering model
    kmeans_clustering = kmeans_clustering_op(input_path = tensorflow_training.outputs['embeddings_output'])

    # Save clustering model in PV
    clustering_model_storing = model_storing_op(model_path = kmeans_clustering.output, minio_url = minio_url,
        minio_access_key = minio_access_key, minio_secret_key = minio_secret_key, minio_path = minio_filename)

    # Start Graph Seldon Deployment
    seldon_deploy = kfp.dsl.ResourceOp(
        name = "Seldon Deploy",
        k8s_resource = deployment_dict,
        action = 'apply'
    ).after(encoder_model_storing, clustering_model_storing)

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(enhanced_locus_pipeline, __file__ + '.tar.gz')
