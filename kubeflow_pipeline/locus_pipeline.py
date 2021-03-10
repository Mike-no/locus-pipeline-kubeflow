import kfp
from kfp import dsl

def pandas_loading_op(file_path):
    return dsl.ContainerOp(
        name = 'Pandas Loading',
        image = 'docker.io/mikeno/pandas_loading:latest',
        command = [ 'python3', '/pandas_loading.py' ],
        arguments = [ '-f', file_path ],
        file_outputs = { 'output': '/data.npz' }
    )

def numpy_preprocessing_op(file_path):
    return dsl.ContainerOp(
        name = 'Numpy Preprocessing',
        image = 'docker.io/mikeno/numpy_preprocessing:latest',
        command = [ 'python3', '/numpy_preprocessing.py' ],
        arguments = [ '-f', file_path ],
        file_outputs = { 'output': '/preprocessed_data.npy' }
    )

@dsl.pipeline(name = 'Locus Pipeline', description = 'Locus Pipeline Encoder + Clustering')
def locus_pipeline(file_path):
    pandas_loading = pandas_loading_op(file_path)

    data = dsl.InputArgumentPath(pandas_loading.output)
    numpy_preprocessing = numpy_preprocessing_op(data)

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(locus_pipeline, __file__ + '.tar.gz')
