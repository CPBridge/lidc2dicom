import argparse, sys, shutil, os, logging
import qt, ctk, slicer
import DICOMLib
from DICOMLib.DICOMUtils import TemporaryDICOMDatabase
import logging

class SlicerLIDCLoader():
  @property
  def dicomDatabase(self):
    return slicer.dicomDatabase

  @property
  def patients(self):
    p = self.dicomDatabase.patients()
    print(p)
    return p

  def __init__(self, tempDB=None):
    self.tempDICOMDB = tempDB

  def importDirectory(self, inputDir):
    print('Input directory: %s' % inputDir)
    self.indexer = getattr(self, "indexer", None)
    if not self.indexer:
      self.indexer = ctk.ctkDICOMIndexer()
    self.indexer.addDirectory(self.dicomDatabase, inputDir)
    print('Import completed, total %s patients imported' % len(self.patients))

  def loadSeries(self, seriesInstanceUIDToLoad=None, modalityToLoad=None, seriesDescriptionPrefix=None):
    for patient in self.patients:
      print(patient)
      for study in self.dicomDatabase.studiesForPatient(patient):
        #print self.dicomDatabase.seriesForStudy(study)

        series = self.dicomDatabase.seriesForStudy(study)
        for seriesIndex, currentSeries in enumerate(series, start=1):
          files = self.dicomDatabase.filesForSeries(currentSeries)

          if len(files) == 0:
            continue

          thisSeriesInstanceUID = self.dicomDatabase.fileValue(files[0], '0020,000e')
          seriesDescription = self.dicomDatabase.fileValue(files[0], '0008,103E')
          print("Found series "+seriesDescription)
          modality = self.dicomDatabase.fileValue(files[0], '0008,0060')

          if seriesInstanceUIDToLoad and modality != "CT":
            if thisSeriesInstanceUID != seriesInstanceUIDToLoad:
              print("Skipping because of series UID not matching "+seriesInstanceUIDToLoad)
              continue

          if modalityToLoad and modality != "CT":
            if modality != modalityToLoad:
              print("Skipping because of modality")
              continue

          if seriesDescriptionPrefix and modality != "CT":
            if not seriesDescription.startswith(seriesDescriptionPrefix):
              print("Skipping series "+seriesDescription)
              continue

          print("Found series of modality "+modality)
          if modality == "CT":
            plugin, loadable = self._getPluginAndLoadableForFiles(seriesDescription, files, ["DICOMScalarVolumePlugin"])
            plugin.load(loadable)
          elif modality == "SR":
            plugin, loadable = self._getPluginAndLoadableForFiles(seriesDescription, files, ["DICOMTID1500Plugin"])

            plugin.load(loadable)

    return True

  def _getPluginAndLoadableForFiles(self, seriesDescription, files, plugins=[]):
    for pluginName in plugins:
      plugin = slicer.modules.dicomPlugins[pluginName]()
      loadables = plugin.examine([files])
      if len(loadables) == 0:
        print("No loadables")
        continue
      loadables.sort(key=lambda x: x.confidence, reverse=True)
      if loadables[0].confidence > 0.1:
        print("Have confident loadable!")
        return plugin, loadables[0]
    return None, None

  def showSegmentations(self):

    from QRCustomizations import CustomSegmentEditor
    import vtkSegmentationCorePython
    vtkSegConverter = vtkSegmentationCorePython.vtkSegmentationConverter
    snc = slicer.mrmlScene.GetNodesByClass("vtkMRMLSegmentationNode")
    for snNumber in range(snc.GetNumberOfItems()):

      sn = snc.GetItemAsObject(snNumber)
      segmentation = sn.GetSegmentation()
      segmentation.SetConversionParameter('Smoothing factor', '0.0')
      segmentation.CreateRepresentation(vtkSegConverter.GetSegmentationClosedSurfaceRepresentationName(), True)

      csl=CustomSegmentEditor.CustomSegmentEditorLogic()
      segmentNode = segmentation.GetNthSegment(0)
      centroid = csl.getSegmentCentroid(sn, segmentNode)
      markupsLogic = slicer.modules.markups.logic()
      markupsLogic.JumpSlicesToLocation(centroid[0],centroid[1],centroid[2], True)

    # center 3d viewer on the segmentation surface
    t=slicer.app.layoutManager().threeDWidget(0).threeDView().resetFocalPoint()


slicer.app.layoutManager().setLayout(slicer.vtkMRMLLayoutNode.SlicerLayoutFourUpView)

# first two are required!
CT_DICOM_PATH = "<CT_DICOM_PATH_PLACEHOLDER>"
DERIVED_DICOM_PATH = "<DERIVED_DICOM_PATH_PLACEHOLDER>"

SR_SERIES_UID = <SR_SERIES_UID_PLACEHOLDER>
SERIES_DESCRIPTION_PREFIX = <SERIES_DESCRIPTION_PREFIX_PLACEHOLDER>

with TemporaryDICOMDatabase(os.path.join("/Users/fedorov/Temp/SlicerScripts", "CtkDICOMDatabase")) as db:
  srSeries = SR_SERIES_UID
  sl = SlicerLIDCLoader(tempDB=db)
  sl.importDirectory(CT_DICOM_PATH)
  sl.importDirectory(DERIVED_DICOM_PATH)
  if srSeries is None:
    # load all SRs and corresponding SEG+CT
    sl.loadSeries(modalityToLoad="SR", seriesDescriptionPrefix=SERIES_DESCRIPTION_PREFIX)
  else:
    # load just one SR
    sl.loadSeries(seriesInstanceUIDToLoad=srSeries, seriesDescriptionPrefix=SERIES_DESCRIPTION_PREFIX)

  # make all segmentations visible in slice viewers and 3d
  sl.showSegmentations()
