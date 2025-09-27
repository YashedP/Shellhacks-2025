# -*- coding: utf-8 -*-

"""
/***************************************************************************
 SHOorelineDEtectionModel - SHODEM
                                 A QGIS plugin
 This plug-in enables shoreline detection from an RGB satellite image.
                              -------------------
        begin                : Created on Tue Feb  7 16:43:36 2023
        copyright            : (C) 2023 by Pietro Scala, Giorgio Manno, Giuseppe Ciarolo; University of Palermo
        email                : pietro.scala@you.unipa.it
 ***************************************************************************/

__Code author__ = 'Pietro Scala, University of Palermo'


/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""


from qgis.core import QgsProcessing
from qgis.core import QgsProcessingAlgorithm
from qgis.core import QgsProcessingMultiStepFeedback
from qgis.core import QgsProcessingParameterNumber
from qgis.core import QgsProcessingParameterVectorLayer
from qgis.core import QgsProcessingParameterRasterLayer
from qgis.core import QgsProcessingParameterFeatureSink
import processing


class Rgb_shodem_shoreline_detection_model(QgsProcessingAlgorithm):

    def initAlgorithm(self, config=None):
        # Use:
- 10 for pale sandy beachs
- 50 for dark sandy beachs
- 100 for rock and sandy beach

        self.addParameter(QgsProcessingParameterNumber('insertscale', 'Insert scale number: \n- 10 for pale sandy beachs \n- 50 for dark sandy beachs\n- 100 for rock and sandy beach\n', type=QgsProcessingParameterNumber.Integer, minValue=1, maxValue=100, defaultValue=50))
        # Enter an "area" type geometry containing a buffer of at least 20 meters of an approximate coastline contour.
        self.addParameter(QgsProcessingParameterVectorLayer('insertthecontourbufferhere', 'Insert the contour buffer here', types=[QgsProcessing.TypeVectorPolygon], defaultValue=None))
        # Here you have to insert the 3 band image (i.e. orthophoto). A 1:2500 scale image with a resolution of 0.5 m or less is recommended.
        self.addParameter(QgsProcessingParameterRasterLayer('insertthejpegimagergbbandshere', 'Insert the JPEG Image (RGB bands)  here', defaultValue=None))
        # Insert the minimum segment that probably intersects the shoreline
        self.addParameter(QgsProcessingParameterVectorLayer('inserttheminimumsegmentthatprobablyintersectstheshoreline', 'Insert the intesecting segment ', types=[QgsProcessing.TypeVectorLine], defaultValue=None))
        self.addParameter(QgsProcessingParameterFeatureSink('Shoreline', 'Shoreline', type=QgsProcessing.TypeVectorAnyGeometry, createByDefault=True, defaultValue=None))
        self.addParameter(QgsProcessingParameterFeatureSink('NeuralNetworkMaskForSemanticSegmentation', 'Neural Network mask for semantic segmentation', type=QgsProcessing.TypeVectorAnyGeometry, createByDefault=True, supportsAppend=True, defaultValue=None))

    def processAlgorithm(self, parameters, context, model_feedback):
        # Use a multi-step feedback, so that individual child algorithm progress reports are adjusted for the
        # overall progress through the model
        feedback = QgsProcessingMultiStepFeedback(4, model_feedback)
        results = {}
        outputs = {}

        # Cutting image for faster simulation
        alg_params = {
            'ALPHA_BAND': False,
            'CROP_TO_CUTLINE': False,
            'DATA_TYPE': 0,  # Usa Il Tipo Dati del Layer in Ingresso
            'EXTRA': '',
            'INPUT': parameters['insertthejpegimagergbbandshere'],
            'KEEP_RESOLUTION': True,
            'MASK': parameters['insertthecontourbufferhere'],
            'MULTITHREADING': False,
            'NODATA': None,
            'OPTIONS': '',
            'OUTPUT': 'TEMPORARY_OUTPUT',
            'SET_RESOLUTION': False,
            'SOURCE_CRS': 'ProjectCrs',
            'TARGET_CRS': 'ProjectCrs',
            'X_RESOLUTION': None,
            'Y_RESOLUTION': None,
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['CuttingImageForFasterSimulation'] = processing.run('gdal:cliprasterbymasklayer', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(1)
        if feedback.isCanceled():
            return {}

        # Segmented pixel for the detection
        alg_params = {
            'BAND': 1,
            'CREATE_3D': False,
            'EXTRA': '',
            'FIELD_NAME': 'Line_Type',
            'IGNORE_NODATA': False,
            'INPUT': outputs['CuttingImageForFasterSimulation']['OUTPUT'],
            'INTERVAL': parameters['insertscale'],
            'NODATA': None,
            'OFFSET': 0,
            'OUTPUT': 'TEMPORARY_OUTPUT',
            'OUTPUT': QgsProcessing.TEMPORARY_OUTPUT
        }
        outputs['SegmentedPixelForTheDetection'] = processing.run('gdal:contour', alg_params, context=context, feedback=feedback, is_child_algorithm=True)

        feedback.setCurrentStep(2)
        if feedback.isCanceled():
            return {}

        # Extract for position of the probably shoreline
        alg_params = {
            'INPUT': outputs['SegmentedPixelForTheDetection']['OUTPUT'],
            'INTERSECT': parameters['inserttheminimumsegmentthatprobablyintersectstheshoreline'],
            'PREDICATE': [4,0,7],  # tocca,interseca,attraversa
            'OUTPUT': parameters['Shoreline']
        }
        outputs['ExtractForPositionOfTheProbablyShoreline'] = processing.run('native:extractbylocation', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['Shoreline'] = outputs['ExtractForPositionOfTheProbablyShoreline']['OUTPUT']

        feedback.setCurrentStep(3)
        if feedback.isCanceled():
            return {}

        # Convert for semantic segmentation mask layer
        alg_params = {
            'INPUT': outputs['ExtractForPositionOfTheProbablyShoreline']['OUTPUT'],
            'OUTPUT': parameters['NeuralNetworkMaskForSemanticSegmentation']
        }
        outputs['ConvertForSemanticSegmentationMaskLayer'] = processing.run('qgis:linestopolygons', alg_params, context=context, feedback=feedback, is_child_algorithm=True)
        results['NeuralNetworkMaskForSemanticSegmentation'] = outputs['ConvertForSemanticSegmentationMaskLayer']['OUTPUT']
        return results

    def name(self):
        return 'RGB_SHODEM_SHOreline_DEtection_Model'

    def displayName(self):
        return 'RGB_SHODEM_SHOreline_DEtection_Model'

    def group(self):
        return ''

    def groupId(self):
        return ''

    def createInstance(self):
        return Rgb_shodem_shoreline_detection_model()



from tensorflow.keras import backend as K
from tensorflow import keras
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np 
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import os
from qgis.core import *

directory = os.getcwd()
print(directory)



def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0)/(K.sum(y_true_f) + K.sum(y_pred_f) - intersection +1.0)

modello = keras.models.load_model("OUTPUT\\SHODEM_NN",custom_objects={'jacard_coef':jacard_coef},compile=False)


modello.summary()
#path_attuale=QgsProject.instance().homePath();
#os.chdir(path_attuale)

basewidth = 512
img = Image.open('OUTPUT\\JPG.jpg')
wpercent = (basewidth / float(img.size[0]))
hsize = 512
img = img.resize((basewidth, hsize), Image.ANTIALIAS)
img.save('OUTPUT\\JPG_r.png')
im = Image.open('OUTPUT\\JPG_r.png')
rgb_im = im.convert('RGB')
rgb_im.save('OUTPUT\\JPG_res.jpg')

test_img = cv2.imread('OUTPUT\\JPG_res.jpg')

#test_img.shape
test_img_input=np.expand_dims(test_img, 0)
prediction = (modello.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]
data=predicted_img

plt.figure()
imgplot = plt.imshow(test_img)
plt.show()

plt.figure()
imgplot = plt.imshow(predicted_img)
plt.show()