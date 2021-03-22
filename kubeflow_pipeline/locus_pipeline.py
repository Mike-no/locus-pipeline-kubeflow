import os
import kfp

# Components 
downloader_op = kfp.components.load_component_from_file(os.path.join('./pandas_loading/downloader/', 'component.yaml'))
pandas_loading_op = kfp.components.load_component_from_file(os.path.join('./pandas_loading/', 'component.yaml'))
numpy_preprocessing_op = kfp.components.load_component_from_file(os.path.join('./numpy_preprocessing/', 'component.yaml'))

@kfp.dsl.pipeline(
    name = 'Pandas Pipeline',
    description = 'Locus Pipeline (Pandas Loading)'
)
def pandas_pipeline(data_url):
    downloader = downloader_op(url = data_url)
    pandas_loading = pandas_loading_op(input_path = downloader.output)
    numpy_preprocessing = numpy_preprocessing_op(input_path = pandas_loading.output)

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(pandas_pipeline, __file__ + '.tar.gz')