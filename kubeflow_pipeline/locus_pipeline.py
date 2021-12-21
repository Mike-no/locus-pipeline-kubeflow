import os
import kfp
import yaml

# Components 
downloader_op = kfp.components.load_component_from_file(os.path.join('./pandas_loading/downloader/', 'component.yaml'))
pandas_loading_op = kfp.components.load_component_from_file(os.path.join('./pandas_loading/', 'component.yaml'))
numpy_preprocessing_op = kfp.components.load_component_from_file(os.path.join('./numpy_preprocessing/', 'component.yaml'))
tensorflow_training_op = kfp.components.load_component_from_file(os.path.join('./tensorflow_training/', 'component.yaml'))
model_storing_op = kfp.components.load_component_from_file(os.path.join('./model_storing/', 'component.yaml'))
kmeans_clustering_op = kfp.components.load_component_from_file(os.path.join('./kmeans_clustering/', 'component.yaml'))

@kfp.dsl.pipeline(
    name = 'UC1 Functionality-1',
    description = 'Locus Pipeline for UC1 Functionality-1'
)
def locus_pipeline(data_url, minio_url, minio_access_key, minio_secret_key):
    
    # Download dataset
    downloader_dataset = downloader_op(url = data_url)

    # Get graph deployment file and 
    deployment_dict = yaml.safe_load(open('../graph/graph.yaml', 'r'))
    minio_path = deployment_dict["spec"]["predictors"][0]["graph"]["modelUri"].split("s3://")[1]
    minio_filename = deployment_dict["spec"]["predictors"][0]["componentSpecs"][0]["spec"]["containers"][0]["image"].split("/")[1].split(":")[0]
    
    # Preprocess and Train encoder model, produce embeddings
    pandas_loading = pandas_loading_op(input_path = downloader_dataset.output)
    numpy_preprocessing = numpy_preprocessing_op(input_path = pandas_loading.output)
    tensorflow_training = tensorflow_training_op(input_path = numpy_preprocessing.output)
    
    # Save encoder model in PV
    encoder_model_storing = model_storing_op(model_path = tensorflow_training.outputs['output_path'], minio_url = minio_url, 
        minio_access_key = minio_access_key, minio_secret_key = minio_secret_key, minio_path = minio_path)
    
    # Produce clustering model
    kmeans_clustering = kmeans_clustering_op(input_path = tensorflow_training.outputs['output_path2'])
    
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
    compiler.Compiler().compile(locus_pipeline, __file__ + '.tar.gz')