# Hola!
Para esta demo he seleccionado dos proyectos donde mejor se puede ver mis habilidades de desarrollo

## FreeWillAI
FreeWillAI es una plataforma que permite deployar modelos de IA en la blockchain.
Este proyecto demuestra mis habilidades de desarrollo de producto y/o MVP, manejo de distintos lenguages de programacion (python, solidity, bash), versatilidad, testeos, creatividad y agilidad.

Piense que aca cree un producto donde no lo habia, cada paso fue super importante y cada feature fue esencial.

## KubeFlow demostration
> [!NOTE]
> Si quiere ver el codigo, tenga en cuenta la carpeta `kubeflow-demo/pipes_code` donde aqui yacen dichas pipelines una que corre en kubeflow `kubeflow-demo/pipes_code/src/kubeflow` (necesitara un cluster de kubernetes con kubeflow instalado) y otra en local `kubeflow-demo/pipes_code/src/local`.
Este codigo lo escribi con un enfoque didactico para aprender de una nueva feature de kubeflow llamada [Lightweight Python Components](https://www.kubeflow.org/docs/components/pipelines/v2/components/lightweight-python-components/).
Este proyecto me motivo a hacer una pipeline de reentrenamiento de modelos IA demostrando mi habilidad para escribir codigo escalable, duradero y facil de debuggear

## Caso UMA
Hablando con Gustavo se me ocurrian varias arquitecturas que se pueden usar para escalar el codigo detro de UMA. Por eso, traje el codigo de kubeflow a este repo.
Igualmente el codigo en la carpeta kubeflow es un poco extenso y cada componente se puede generalizar para un uso mas abarcativo.

### Ejemplos

Aqui esta como definiria los componentes en cada pipeline, tomare de ejemplo el componente de preprocesamiento.
```python
@dataclass
class PreprocessInput:
    user_data: DatasetArtifact


@dataclass
class PreprocessOutput:
    train_dataset: DatasetArtifact
    test_dataset: DatasetArtifact
    val_dataset: DatasetArtifact


class PreprocessComponent(Component):
    name = 'preprocess'

    @classmethod
    def do_preprocess(cls, input: PreprocessInput, ctx: Execution) -> PreprocessOutput:
        raise NotImplementedError

    @classmethod
    def preprocess(cls, input: PreprocessInput) -> PreprocessOutput:
        with Execution(name=cls.name, metadata=cls.metadata) as ctx:
            return cls.do_preprocess(input, cast(Execution, ctx))
```
En estas simples abstracciones tenemos todo lo que necesitamos para cada componente: Tener bien declarados los datos de entrada y de salida, ademas de un control del contexto donde estaran alojados los metadatos y configuraciones necesarias para control de flujo o lo que sea que el desarrollador quisiera implementar para su rapida implementacion.

otro ejemplo
```python
@dataclass
class TrainInput:
    train_dataset: DatasetArtifact
    val_dataset: DatasetArtifact


@dataclass
class TrainOutput:
    model: ModelArtifact


class TrainComponent(Component):
    name = 'train'

    @classmethod
    def do_train(cls, input: TrainInput, ctx: Execution) -> TrainOutput:
        raise NotImplementedError

    @classmethod
    def train(cls, input: TrainInput) -> TrainOutput:
        with Execution(name=cls.name, metadata=cls.metadata) as ctx:
            return cls.do_train(input, cast(Execution, ctx))
```

Armemos el flujo de nuestra pipeline de reentrenamiento utilizando cada uno de estos componentes y ya lo tendriamos
```python
class Pipeline():
    def __init__(self, input: PipelineInput, cfg: PipelineConfig):
        self.cfg = cfg
        self.input = input

    def run(self) -> PipelineOutput | None:
        raise NotImplementedError


class BaseReTrainPipeline(
    Pipeline,
    IngestComponent,
    PreprocessComponent,
    TrainComponent,
    EvaluationComponent,
    DeployComponent,
):
    def run(self) -> None:
        ingest_op = self.ingest(self.input)

        preproc_in = PreprocessInput(ingest_op.base_data, ingest_op.user_data)
        preproc_op = self.preprocess(preproc_in)

        train_in = TrainInput(preproc_op.train_dataset, preproc_op.val_dataset)
        train_op = self.train(train_in)
        
        eval_in = EvaluationInput(train_op.model, ingest_op.base_model, preproc_op.test_dataset)
        eval_op = self.eval(eval_in)

        results_dict = eval_op.results.content()
        if results_dict['current_model_loss'] < results_dict['base_model_loss']:
            deploy_in = DeployInput(ingest_op.user_data, ingest_op.base_data, train_op.model)
            deploy_op = self.deploy(deploy_in)

            if not deploy_op.deployed:
                raise RuntimeError('Error trying to deploy')
```
Lo unico que debemos hacer para correrla es declarar una clase `PipelineInput` con los datos necesarios y correr el metodo `run`.
Asi se ve el codigo para una tarea especifica (tomando el ejemplo de los diamantes del repo de kubeflow)
```python
class DiamondsLocalPipeline(
    BaseReTrainPipeline,
    LocalIngest,
    DiamondsPreprocess,
    DiamondsTrain,
    DiamondsEvaluation,
    LocalDeploy
):
    def __init__(self, input: DiamondsPipelineInput, cfg = DiamondsDatasetConfig()):
        self.input = input
        self.cfg = cfg
        self.__class__.cfg = cfg
```
Imaginemos que queremos utilizar un bucket de S3 con imagenes, reentrenar un modelo que prediga si un paciente tiene cancer de pulmon y deployarlo usando github actions. La pipeline se veria algo asi:
```python
class LungCancerReTrainPipeline(
    BaseReTrainPipeline,
    S3Ingestor,
    ImagePreprocessor,
    LungCancerTrainer,
    VisionEvaluator,
    GithubDeploy
):
    def __init__(self, input: LungCancerPipelineInput, cfg = LungCancerDatasetConfig()):
        self.input = input
        self.cfg = cfg
        self.__class__.cfg = cfg
```
Cuanto mas componentes, el cientifico y/o ingeniero tiene que escribir menos codigo y puede iterar mas rapido.

Y muy importante, esto se puede replicar para correr agentes, al final son componentes con dependencias y flujo de datos.


## BONUS: Como correr freewillai
> [!IMPORTANT]
> Este producto tiene muchas dependecias a la hora de correrlo y por decision del equipo no hicimos un dockerfile acorde (teniamos nuestro propio rig de mineria que sirvio como server y todo lo manejamos desde alli).
> Por eso aqui le dejo todo lo que debe de instalar antes de correrlo:
> 
> Foundry, este es el framework mas usado para el desarrollo en web3, comodo y escalable
>   link: https://book.getfoundry.sh/getting-started/installation
>
> IPFS (Inter Planetary File System), Este es el sistema de archivos descentralizado mas grande que existe y donde freewillai alojara sus modelos y datos
>   link: https://docs.ipfs.tech/install/command-line/

Debe de estar dentro del directorio para todos los siguientes pasos
```bash
cd freewillai
```

Levantar Anvil (Anvil es un endpoint de ethereum que corre de manera local y funciona como un entorno de pruebas)
> [!NOTE]
> Tenga a mano el output de este comando porque necesitara una de las claves privadas que se muestran
```bash
bash scripts/run_anvil.sh
```

Levantar un nodo de IPFS
```bash
bash scripts/run_ipfs.sh
```

Correr al menos dos nodos de freewillai
```bash
bash scripts/run_node.sh <pege-aqui-una-clave-privada-de-anvil>
bash scripts/run_node.sh <pege-aqui-otra-clave-privada-de-anvil>
```

Genial! ahora tenemos una red de freewillai tan grande como 2 nodos validadores

Como correr AI en freewillai usando python
```bash
echo "PRIVATE_KEY=<pega-aqui-una-nueva-clave-privada-de-anvil>" >> .env
```
```python
# Ejemplo usando un modelo basico en keras
import freewillai
import keras

model_path = 'bucket/test/models/keras_model_dnn/'
model = keras.models.load_model(model_path)
dataset = 'bucket/test/datasets/keras_testing_dataset.csv'

freewillai.connect("devnet/anvil", env_file='.env')
result = await freewillai.run_task(model, dataset)

print(result)
```
