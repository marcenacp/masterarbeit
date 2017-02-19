# Master's thesis

*-- Simultaneous localization and classification of environmental sounds using deep learning techniques --*

## Abstract

The purpose of the present thesis is the simultaneous classification and localization of
sounds using bio-inspired computing, including deep learning and binaural recording.
This falls within the scope of computational auditory scene analysis, where a machine-
hearing agent shall identify and localize sounds from its environment.
The data is a mixture of binaural recordings related to a fire emergency (alarm, fire
noise, people shouting, etc.) along with their related labels (for classification) and
azimuths (for localization). The extracted features that are fed into the machine-
hearing agent are similar to these used by the actual auditory system, such as
ratemaps.

Representing sounds mainly by their spectrogram images allows us to train our
data on convolutional neural networks, which appear to be particularly suitable for
pattern recognition on images. Appropriate metrics help select and assess the best
architectures along with a random search on the hyperparameter space.

Clean sounds can be highly accurately identified (88.8% accuracy) and localized
(88.7% accuracy). Although mixed sounds identification is also achieved (71.0%
balanced accuracy for the best class), their localization has proved a much harder
task (60.0% balanced accuracy for the best azimuth). Side conclusions can also be
drawn with respect to feature transferability in deep networks, supporting the fact
that layers become increasingly more task-specific as depth increases.

**Keywords**: bio-inspired artificial intelligence, deep learning, convolutional neural
networks, supervised classification, multitask learning, machine-hearing system.
