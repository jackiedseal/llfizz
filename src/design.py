from faidr.libs.featurize import Featurize
from faidr.core.dataclasses import ProteinSequence
from faidr.libs.design.randmch import PointMutMinimizeNorm

DED1 = "MMSAPGSRSNSRRGGFGRNNNRDYRKAGGASAGGWGSSRSRDNSFRGGSGWGSDSKSSGWGNSGGSNNSSWW"
COX15 = "MLFRNIEVGRQAAKLLTRTSSRLAWQSIGASRNISTIRQQIRKTQ"

def main():
    featurize = Featurize()
    seq = ProteinSequence(DED1)
    feature_vector = featurize(seq)

    print(feature_vector)

    optimizer = PointMutMinimizeNorm()
    designed_sequence = optimizer.fit_random_to_target(target=DED1)