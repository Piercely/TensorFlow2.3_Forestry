from flask import Flask, render_template, request
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

app = Flask(__name__)

# 模型初始化
model = tf.keras.models.load_model("models/mobilenet_Forestry.h5")  # 加载模型
class_names = ['ActiasDubernardiOberthur', 'ActiasSeleneNingpoanaFelder', 'AgriusConvolvuli', 'AmsactaLactinea',
                    'AnoplophoraChinensisForster', 'AnoplophoraGlabripennisMotschulsky', 'AprionaGermari',
                    'AprionaSwainsoni', 'AromiaBungiiFald', 'AtaturaIlia', 'BatoceraHorsfieldiHope',
                    'ByasaAlcinousKlug', 'CalospilosSuspectaWarren', 'CamptolomaInteriorata',
                    'CarposinaNiponensisWalsingham', 'CatharsiusMolossusLinnaeus', 'CeruraMencianaMoore',
                    'ChalcophoraJaponica', 'CicadellaViridis', 'ClanisBilineata', 'CletusPunctigerDallas',
                    'ClosteraAnachoreta', 'ClosteraAnastomosis', 'ConogethesPunctiferalis', 'CorythuchaCiliata',
                    'CreatonotusTransiens', 'CryptotympanaAtrataFabricius', 'CyclidiaSubstigmariaSubstigmaria',
                    'CyclopeltaObscura', 'CystidiaCouaggariaGuenee', 'DanausChrysippusLinnaeus', 'DanausGenutia',
                    'DasychiraGroteiMoore', 'DendrolimusPunctatusWalker', 'DiaphaniaPerspectalis',
                    'DicranocephalusWallichi', 'DictyopharaSinica', 'DorcusTitanusPlatymelus', 'DrosichaCorpulenta',
                    'EligmaNarcissus', 'EnmonodiaVespertiliFabricius', 'ErthesinaFullo', 'EuricaniaClara',
                    'EurydemaDominulus', 'GeishaDistinctissima', 'GraphiumSarpedonLinnaeue', 'GraphosomaRubrolineata',
                    'HalyomorphaPicusFabricius', 'HestinaAssimilis', 'HistiaRhodopeCramer', 'HyphantriaCunea',
                    'JacobiascaFormosana', 'LatoriaConsociaWalker', 'LethocerusDeyrolliVuillefroy',
                    'LocastraMuscosalisWalker', 'LycormaDelicatula', 'MegopisSinicaSinicaWhite', 'MeimunaMongolica',
                    'MicromelalophaTroglodyta', 'MiltochristaStriata', 'MonochamusAlternatusHope',
                    'Ophthalmitisirrorataria', 'OrthagaAchatina', 'PapilioBianorCramer', 'PapilioMachaonLinnaeus',
                    'PapilioPolytesLinnaeus', 'PapilioProtenorCramer', 'PapilioXuthusLinnaeus', 'ParocneriaFurva',
                    'PergesaElpenorlewisi', 'PidorusAtratusButter', 'PierisRapae', 'PlagioderaVersicolora',
                    'PlatypleuraKaempferi', 'PlinachtusBicoloripesScott', 'PlinachtusDissimilis', 'PolygoniaCaureum',
                    'PolyuraNarcaeaHewitson', 'PorthesiaSimilis', 'ProdeniaLitura', 'ProtaetiaBrevitarsisLewis',
                    'PsilogrammaMenephron', 'RicaniaSublimata', 'RiptortusPedestris', 'SemanotusBifasciatusBifasciatus',
                    'SericinusMontelusGrey', 'SinnaExtrema', 'SmerinthusPlanusWalker', 'SpeiredoniaRetorta',
                    'SpilarctiaRobusta', 'SpilarctiaSubcarnea', 'StilprotiaSalicis', 'TheretraJaponica',
                    'ThoseaSinensisWalker', 'UropyiaMeticulodina', 'VanessaIndicaHerbst']  # todo 修改类名，这个数组在模型训练的开始会输出
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    img = Image.open(file)
    img = np.asarray(img)
    img = cv2.resize(img, (224, 224))

    outputs = model.predict(img.reshape(1, 224, 224, 3))
    result_index = int(np.argmax(outputs))
    result = class_names[result_index]

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
