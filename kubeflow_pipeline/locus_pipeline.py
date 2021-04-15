import os
import kfp

# Components 
downloader_op = kfp.components.load_component_from_file(os.path.join('./pandas_loading/downloader/', 'component.yaml'))
pandas_loading_op = kfp.components.load_component_from_file(os.path.join('./pandas_loading/', 'component.yaml'))
numpy_preprocessing_op = kfp.components.load_component_from_file(os.path.join('./numpy_preprocessing/', 'component.yaml'))
tensorflow_training_op = kfp.components.load_component_from_file(os.path.join('./tensorflow_training/', 'component.yaml'))
kmeans_clustering_op = kfp.components.load_component_from_file(os.path.join('./kmeans_clustering/', 'component.yaml'))

@kfp.dsl.pipeline(
    name = 'Pandas Pipeline',
    description = 'Locus Pipeline (Pandas Loading)'
)
def pandas_pipeline(data_url):
    downloader = downloader_op(url = data_url)
    pandas_loading = pandas_loading_op(input_path = downloader.output)
    numpy_preprocessing = numpy_preprocessing_op(input_path = pandas_loading.output)
    tensorflow_training = tensorflow_training_op(input_path = numpy_preprocessing.output)
    kmeans_clustering = kmeans_clustering_op(input_path = tensorflow_training.outputs['output_path2'])

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(pandas_pipeline, __file__ + '.tar.gz')