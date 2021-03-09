import kfp
from kfp import dsl

def pandas_loading_op(file_path):
    return dsl.ContainerOp(
        name = 'Pandas Loading',
        image = 'docker.io/mikeno/pandas_loading:latest',
        command = [ 'python3', '/pandas_loading.py' ],
        arguments = [ '-f', file_path ],
        file_outputs = { 'output': '/processed_data.npy' }
    )

@dsl.pipeline(name = 'Locus Pipeline', description = 'Locus Pipeline Encoder + Clustering')
def locus_pipeline(file_path):
    pandas_loading = pandas_loading_op(file_path)

if __name__ == '__main__':
    import kfp.compiler as compiler
    compiler.Compiler().compile(locus_pipeline, __file__ + '.tar.gz')
