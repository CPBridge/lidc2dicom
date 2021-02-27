from pathlib import Path
import lidc_conversion_utils.helpers as lidc_helpers
import os, itk, tempfile, json, pydicom, tempfile, shutil, sys
import subprocess
import pylidc as pl
import numpy as np
import glob
import logging
from decimal import *

from pydicom.sr.codedict import codes
from pydicom.uid import generate_uid

from highdicom.version import __version__ as highdicom_version
from highdicom.content import AlgorithmIdentificationSequence
from highdicom.seg.content import SegmentDescription
from highdicom.seg.enum import SegmentAlgorithmTypeValues, SegmentationTypeValues
from highdicom.seg.sop import Segmentation

from highdicom.sr.content import (
    FindingSite,
    ImageRegion3D,
    SourceImageForMeasurement
)
from highdicom.sr.enum import GraphicTypeValues3D
from highdicom.sr.sop import Comprehensive3DSR
from highdicom.sr.templates import (
    AlgorithmIdentification,
    DeviceObserverIdentifyingAttributes,
    Measurement,
    MeasurementProperties,
    MeasurementReport,
    ReferencedSegment,
    ObservationContext,
    ObserverContext,
    PersonObserverIdentifyingAttributes,
    VolumetricROIMeasurementsAndQualitativeEvaluations,
    TrackingIdentifier,
)
from highdicom.sr.content import SourceSeriesForSegmentation
from highdicom.sr.value_types import CodedConcept, CodeContentItem


class LIDC2DICOMConverter:

  def __init__(self, args):
    self.logger = logging.getLogger("lidc2dicom")

    self.args = args
    self.rootDir = args.imagesDir
    self.tempDir= args.outputDir

    self.srTemplate = "sr_conversion_template.json"
    self.colorsFile = "GenericColors.txt"

    # read GenericColors
    self.colors = []
    with open(self.colorsFile,'r') as f:
      for l in f:
        if l.startswith('#'):
          continue
        self.colors.append([int(c) for c in l.split(' ')[2:5]])

    self.conceptsDictionary = {}
    self.valuesDictionary = {}
    with open("concepts_dict.json") as cf:
      self.conceptsDictionary = json.load(cf)
    with open("values_dict.json") as vf:
      self.valuesDictionary = json.load(vf)

  def cleanUpTempDir(self, dir):
    for p in Path(dir).glob("*.nrrd"):
      p.unlink()

  #def saveAnnotationAsNRRD(self, annotation, refVolume, fileName):
  #  maskArray = annotation.boolean_mask(10000).astype(np.int16)

  #  maskArray = np.swapaxes(maskArray,0,2).copy()
  #  maskArray = np.rollaxis(maskArray,2,1).copy()

  #  maskVolume = itk.GetImageFromArray(maskArray)
  #  maskVolume.SetSpacing(refVolume.GetSpacing())
  #  maskVolume.SetOrigin(refVolume.GetOrigin())
  #  writerType = itk.ImageFileWriter[itk.Image[itk.SS, 3]]
  #  writer = writerType.New()
  #  writer.SetFileName(fileName)
  #  writer.SetInput(maskVolume)
  #  writer.SetUseCompression(True)
  #  writer.Update()

  def convertSingleAnnotation(self, nCount, aCount, a, ct_datasets, noduleUID, seriesDir, scan):

    # update as necessary!
    noduleName = "Nodule "+str(nCount+1)
    segName = "Nodule "+str(nCount+1) +" - Annotation " + a._nodule_id

    seg_desc = SegmentDescription(
        segment_number=1,
        segment_label=segName,
        segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
        segmented_property_type=codes.SCT.Nodule,
        algorithm_type=SegmentAlgorithmTypeValues.MANUAL,
        algorithm_identification=None,  # TODO
        tracking_uid=noduleUID,
        tracking_id=noduleName,
        anatomic_regions=[codes.SCT.Lung],
    )
    seg_desc.SegmentDescription = segName  # TODO should this be part of the init?
    seg_desc.RecommendedDisplayCIELabValue = self.colors[aCount + 1]  # TODO should this be part of the init?

    self.instanceCount = self.instanceCount+1
    if ct_datasets[0].SeriesNumber != '':
      series_num = int(ct_datasets[0].SeriesNumber) + self.instanceCount
    else:
      series_num = self.instanceCount

    dcmSegFile = os.path.join(self.tempSubjectDir,segName+'.dcm')

    self.logger.info("Converting to DICOM SEG")

    # Construct an enpty mask the same size as the input series
    image_size = (ct_datasets[0].Rows, ct_datasets[0].Columns, len(ct_datasets))
    mask = np.zeros(image_size, np.uint8)

    # Fill in the mask elements with the segmentation
    mask[a.bbox()] = a.boolean_mask().astype(np.int8)

    # Find the subset of the source images relevant for the segmentation
    ct_subset = ct_datasets[a.bbox()[2]]
    mask_subset = mask[(slice(None), slice(None), a.bbox()[2])]
    mask_subset = np.moveaxis(mask_subset, 2, 0)

    seg_dcm = Segmentation(
        source_images=ct_subset,
        pixel_array=mask_subset,
        segmentation_type=SegmentationTypeValues.BINARY,
        segment_descriptions=[seg_desc],
        series_instance_uid=generate_uid(),
        series_number=series_num,
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer="highdicom developers",
        manufacturer_model_name="highdicom",
        software_versions=f"{highdicom_version}",
        device_serial_number='1',
        content_description="Lung nodule segmentation",
        content_creator_name="Reader1",
        series_description=f"Segmentation of {segName}"
    )

    # Add in some extra information
    seg_dcm.BodyPartExamined = "Lung"
    seg_dcm.ClincalTrialSeriesID = "Session1"
    seg_dcm.ClincalTrialTimePointID = "1"
    seg_dcm.ClinicalTrialCoordinatingCenterName = "TCIA"
    seg_dcm.ContentLabel = "SEGMENTATION"

    # Save the file
    seg_dcm.save_as(dcmSegFile)

    segUID = None
    ctSeriesUID = None
    try:
      segDcm = pydicom.read_file(dcmSegFile)
      segUID = segDcm.SOPInstanceUID
      ctSeriesUID = segDcm.ReferencedSeriesSequence[0].SeriesInstanceUID
    except:
      self.logger.error("Failed to read Segmentation file")
      return

    srName = segName+" evaluations"

    # be explicit about reader being anonymous
    observer_context = ObserverContext(
        observer_type=codes.DCM.Person,
        observer_identifying_attributes=PersonObserverIdentifyingAttributes(
            name='anonymous'
        )
    )
    observation_context = ObservationContext(
        observer_person_context=observer_context
    )

    self.instanceCount = self.instanceCount+1
    if ct_datasets[0].SeriesNumber != '':
      series_number = str(int(ct_datasets[0].SeriesNumber)+self.instanceCount)
    else:
      series_number = str(self.instanceCount)

    #srJSON["compositeContext"] = [dcmSegFile.split('/')[-1]]  # TODO what is this?
    #srJSON["imageLibrary"] = os.listdir(seriesDir)   # TODO what is this?

    qualitative_evaluations = []
    for attribute in self.conceptsDictionary.keys():
      try:
        qualitative_evaluations.append(
            CodeContentItem(
                name=CodedConcept(**self.conceptsDictionary[attribute]),
                value=CodedConcept(**self.valuesDictionary[attribute][str(getattr(a, attribute))])
            )
        )
      except KeyError:
        self.logger.info("Skipping invalid attribute: "+attribute+': '+str(getattr(a, attribute)))
        continue

    srName = "Nodule "+str(nCount+1) +" - Annotation " + a._nodule_id + " measurements"

    # TODO
    # Describe the image region for which observations were made
    # (in physical space based on the frame of reference)
    #referenced_region = ImageRegion3D(
    #    graphic_type=GraphicTypeValues3D.POLYGON,
    #    graphic_data=np.array([
    #        (165.0, 200.0, 134.0),
    #        (170.0, 200.0, 134.0),
    #        (170.0, 220.0, 134.0),
    #        (165.0, 220.0, 134.0),
    #        (165.0, 200.0, 134.0),
    #    ]),
    #    frame_of_reference_uid=image_dataset.FrameOfReferenceUID
    #)

    # Describe the anatomic site at which observations were made
    finding_sites = [FindingSite(anatomic_location=codes.SCT.Lung)]

    referenced_segment = ReferencedSegment(
        sop_class_uid=seg_dcm.SOPClassUID,
        sop_instance_uid=seg_dcm.SOPInstanceUID,
        segment_number=1,
        source_series=SourceSeriesForSegmentation(
            ct_datasets[0].SeriesInstanceUID
        )
    )

    # Describe the imaging measurements for the image region defined above
    referenced_images = [
        SourceImageForMeasurement(
            referenced_sop_class_uid=ds.SOPClassUID,
            referenced_sop_instance_uid=ds.SOPInstanceUID
        )
        for ds in ct_subset
    ]
    pylidc_algo_id = AlgorithmIdentification(name='pylidc', version=pl.__version__)
    volume_measurement = Measurement(
        name=codes.SCT.Volume,
        tracking_identifier=TrackingIdentifier(uid=generate_uid()),
        value=a.volume,
        unit=codes.UCUM.CubicMillimeter,
        referenced_images=referenced_images,
        algorithm_id=pylidc_algo_id
    )
    diameter_measurement = Measurement(
        name=codes.SCT.Diameter,
        tracking_identifier=TrackingIdentifier(uid=generate_uid()),
        value=a.diameter,
        unit=codes.UCUM.Millimeter,
        referenced_images=referenced_images,
        algorithm_id=pylidc_algo_id
    )
    surface_area_measurement = Measurement(
        name=CodedConcept(value='C0JK', scheme_designator='IBSI', meaning="Surface area of mesh"),
        tracking_identifier=TrackingIdentifier(uid=generate_uid()),
        value=a.surface_area,
        unit=codes.UCUM.SquareMillimeter,
        referenced_images=referenced_images,
        algorithm_id=pylidc_algo_id
    )

    imaging_measurements = [
        VolumetricROIMeasurementsAndQualitativeEvaluations(
            tracking_identifier=TrackingIdentifier(
                uid=noduleUID,
                identifier=noduleName
            ),
            #referenced_region=referenced_region, # TODO
            referenced_segment=referenced_segment,
            finding_type=codes.SCT.Nodule,
            measurements=[volume_measurement, diameter_measurement, surface_area_measurement],
            qualitative_evaluations=qualitative_evaluations,
            finding_sites=finding_sites
        )
    ]
    measurement_report = MeasurementReport(
        observation_context=observation_context,
        procedure_reported=codes.LN.CTUnspecifiedBodyRegion,
        imaging_measurements=imaging_measurements
    )

    # Create the Structured Report instance
    sr_dcm = Comprehensive3DSR(
        evidence=ct_subset,
        content=measurement_report[0],
        series_number=series_number,
        series_instance_uid=generate_uid(),
        sop_instance_uid=generate_uid(),
        instance_number=1,
        manufacturer='highdicom developers',
        manufacturer_model_name='highdicom',
        is_complete=True,
        is_verified=True,
        series_description=srName
    )

    sr_dcm.save_as(dcmSRFile)

    if not os.path.exists(dcmSRFile):
      self.logger.error("Failed to access output SR file for "+s)


  def convertForSubject(self, subjectID):
    s = 'LIDC-IDRI-%04i' % subjectID
    self.logger.info("Processing subject %s" % (s))
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == s)
    self.logger.info(" Found %d scans" % (scans.count()))

    for scan in scans:
      studyUID = scan.study_instance_uid
      seriesUID = scan.series_instance_uid
      seriesDir = Path(scan.get_path_to_dicom_files())
      if not os.path.exists(seriesDir):
        self.logger.error("Files not found for subject "+s)
        return

      try:
        ct_datasets = scan.load_all_dicom_images()
      except:
        logger.error("Failed to read input CT files")
        return

      ok = lidc_helpers.checkSeriesGeometry(str(seriesDir))
      if not ok:
        self.logger.warning("Geometry inconsistent for subject %s" % (s))

      self.tempSubjectDir = os.path.join(self.tempDir,s,studyUID,seriesUID)
      os.makedirs(self.tempSubjectDir, exist_ok=True)

      #scanNRRDFile = os.path.join(self.tempSubjectDir,s+'_CT.nrrd')
      #if not os.path.exists(scanNRRDFile):
      #  # convert
      #  # tempDir = tempfile.mkdtemp()
      #  plastimatchCmd = ['plastimatch', 'convert','--input',seriesDir,'--output-img',scanNRRDFile]
      #  self.logger.info("Running plastimatch with "+str(plastimatchCmd))

      #  sp = subprocess.Popen(plastimatchCmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
      #  (stdout, stderr) = sp.communicate()
      #  self.logger.info("plastimatch stdout: "+stdout.decode('ascii'))
      #  self.logger.warning("plastimatch stderr: "+stderr.decode('ascii'))

      #  self.logger.info('plastimatch completed')
      #  self.logger.info("Conversion of CT volume OK - result in "+scanNRRDFile)
      #else:
      #  self.logger.info(scanNRRDFile+" exists. Not rerunning volume reconstruction.")

      #reader = itk.ImageFileReader[itk.Image[itk.SS, 3]].New()
      #reader.SetFileName(scanNRRDFile)
      #reader.Update()
      #volume = reader.GetOutput()

      #logger.info(volume.GetLargestPossibleRegion().GetSize())

      # now iterate over all nodules available for this subject
      anns = scan.annotations
      self.logger.info("Have %d annotations for subject %s" % (len(anns), s))

      self.instanceCount = 0

      clusteredAnnotationIDs = []

      for nCount,nodule in enumerate(scan.cluster_annotations()):

        noduleUID = pydicom.uid.generate_uid(prefix=None) # by default, pydicom uses 2.25 root

        for aCount,a in enumerate(nodule):

          clusteredAnnotationIDs.append(a.id)

          annotationFileName = "Nodule "+str(nCount+1) +" - Annotation " + a._nodule_id+'.nrrd'
          # self.saveAnnotationAsNRRD(a, volume, os.path.join(self.tempSubjectDir,annotationFileName))

          self.convertSingleAnnotation(nCount, aCount, a, ct_datasets, noduleUID, seriesDir, scan)


      if len(clusteredAnnotationIDs) != len(anns):
        self.logger.warning("%d annotations unaccounted for!" % (len(anns) - len(clusteredAnnotationIDs)))

      for ua in anns:
        if ua.id not in clusteredAnnotationIDs:
          aCount = aCount+1
          nCount = nCount+1
          noduleUID = pydicom.uid.generate_uid(prefix=None)
          self.convertSingleAnnotation(nCount, aCount, ua, ct_datasets, noduleUID, seriesDir, scan)

      #self.cleanUpTempDir(self.tempSubjectDir)

  def makeCompositeObjects(self, subjectID):

    # convert all segmentations and measurements into composite objects
    # 1. find all segmentations
    # 2. read all, append metadata
    # 3. find all measurements
    # 4. read all, append metadata
    import re
    s = 'LIDC-IDRI-%04i' % subjectID
    self.logger.info("Making composite objects for "+s)

    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == s)
    self.logger.info(" Found %d scans" % (scans.count()))

    # cannot just take all segmentation files in a folder, since

    for scan in scans:
      studyUID = scan.study_instance_uid
      seriesUID = scan.series_instance_uid
      seriesDir = scan.get_path_to_dicom_files()
      if not os.path.exists(seriesDir):
        self.logger.error("Files not found for subject "+s)
        return

      dcmFiles = glob.glob(os.path.join(seriesDir,"*.dcm"))
      if not len(dcmFiles):
        logger.error("No DICOM files found for subject "+s)
        return

      firstFile = os.path.join(seriesDir,dcmFiles[0])

      try:
        ctDCM = pydicom.read_file(firstFile)
      except:
        logger.error("Failed to read input file "+firstFile)
        return

      self.instanceCount = 1000

      subjectScanTempDir = os.path.join(self.tempDir,s,studyUID,seriesUID)
      allSegmentations = glob.glob(os.path.join(subjectScanTempDir, 'Nodule*Annotation*.nrrd'))

      if not len(allSegmentations):
        continue

      segMetadata = {}
      nrrdSegFileList = ""
      srMetadata = {}

      for segID,seg in enumerate(allSegmentations):

        prefix = seg[:-5]
        matches = re.match('Nodule (\d+) - Annotation (.+)\.', os.path.split(seg)[1])
        print("Nodule: "+matches.group(1)+" Annotation: "+matches.group(2))

        if not segMetadata:
          segMetadata = json.load(open(prefix+".json"))
        else:
          thisSegMetadata = json.load(open(prefix+".json"))
          segMetadata["segmentAttributes"].append(thisSegMetadata["segmentAttributes"][0])

        if not srMetadata:
          srMetadata = json.load(open(prefix+" measurements.json"))
        else:
          thisSRMetadata = json.load(open(prefix+" measurements.json"))
          thisSRMetadata["Measurements"][0]["ReferencedSegment"] = segID+1
          srMetadata["Measurements"].append(thisSRMetadata["Measurements"][0])

        nrrdSegFileList = nrrdSegFileList+seg+","

      segMetadata["ContentDescription"] = "Lung nodule segmentation - all"
      segMetadata["SeriesDescription"] = "Segmentations of all nodules"
      segMetadata["SeriesNumber"] = str(int(ctDCM.SeriesNumber)+self.instanceCount)
      self.instanceCount = self.instanceCount+1

      # run SEG converter

      allSegsJSON = os.path.join(subjectScanTempDir, "all_segmentations.json")
      with open(allSegsJSON,"w") as f:
        json.dump(segMetadata, f, indent=2)

      compositeSEGFileName = os.path.join(subjectScanTempDir,"all_segmentations.dcm")
      nrrdSegFileList = nrrdSegFileList[:-1]

      converterCmd = ['itkimage2segimage', "--inputImageList", nrrdSegFileList, "--inputDICOMDirectory", seriesDir, "--inputMetadata", allSegsJSON, "--outputDICOM", compositeSEGFileName]
      if self.args.skip:
        converterCmd.append('--skip')
      self.logger.info("Converting to DICOM SEG with "+str(converterCmd))

      sp = subprocess.Popen(converterCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      (stdout, stderr) = sp.communicate()
      self.logger.info("itkimage2segimage stdout: "+stdout.decode('ascii'))
      self.logger.warning("itkimage2segimage stderr: "+stderr.decode('ascii'))

      if not os.path.exists(compositeSEGFileName):
        self.logger.error("Failed to access output composite SEG file for "+s)

      # populate composite SR JSON
      # need SEG SOPInstnaceUID for that purpose
      segDcm = pydicom.read_file(compositeSEGFileName)
      segUID = segDcm.SOPInstanceUID
      ctSeriesUID = segDcm.ReferencedSeriesSequence[0].SeriesInstanceUID

      for mItem in range(len(srMetadata["Measurements"])):
        srMetadata["Measurements"][mItem]["segmentationSOPInstanceUID"] = segUID

      srMetadata["compositeContext"] = [os.path.split(compositeSEGFileName)[1]]

      srMetadata["ContentDescription"] = "Lung nodule measurements - all"
      srMetadata["SeriesDescription"] = "Evaluations for all nodules"
      srMetadata["SeriesNumber"] = str(int(ctDCM.SeriesNumber)+self.instanceCount)
      self.instanceCount = self.instanceCount+1

      allSrsJSON = os.path.join(subjectScanTempDir, "all_measurements.json")
      with open(allSrsJSON,"w") as f:
        json.dump(srMetadata, f, indent=2)

      compositeSRFileName = os.path.join(subjectScanTempDir,"all_measurements.dcm")
      nrrdSegFileList = nrrdSegFileList[:-1]

      converterCmd = ['tid1500writer', "--inputMetadata", allSrsJSON, "--inputImageLibraryDirectory", seriesDir, "--inputCompositeContextDirectory", subjectScanTempDir, "--outputDICOM", compositeSRFileName]
      self.logger.info("Converting to DICOM SR with "+str(converterCmd))

      sp = subprocess.Popen(converterCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
      (stdout, stderr) = sp.communicate()
      self.logger.info("tid1500writer stdout: "+stdout.decode('ascii'))
      self.logger.warning("tid1500writer stderr: "+stderr.decode('ascii'))

      if not os.path.exists(compositeSRFileName):
        self.logger.error("Failed to access output composite SR file for "+s)


    #'Nodule (\d+) - Annotation (.*)')
    #print(allSegmentations)

def main():
  import argparse
  parser = argparse.ArgumentParser(
    usage="%(prog)s --subjects <LIDC_subjectID>\n\n"
    "This program will parse the DICOM and XML data for LIDC subject specified and generate"
    "DICOM representation for the segmentations and evaluations of the segmented nodule."
    "More details in a document to follow")
  parser.add_argument(
    '--subject-range',
    dest = "subjectRange",
    nargs=2,
    type=int,
    help = "Range of subject identifiers to be processed. Overrides individual subjects specified."
  )
  parser.add_argument(
    '--all-subjects',
    dest = "allSubjects",
    action="store_true",
    help = "Process all subjects (up to 1012). Overrides all other subject specifications."
  )
  parser.add_argument(
    '--subjects',
    type=int,
    nargs = '+',
    dest="subjectIDs",
    help='Identifier(s) (separated by space) of the subject to be processed.')
  parser.add_argument(
    '--log',
    dest="logFile",
    help="Location of the file to store processing log."
  )
  parser.add_argument(
    '--output-dir',
    dest="outputDir",
    help="Directory for storing the results of conversion."
  )
  parser.add_argument(
    '--composite',
    action="store_true",
    default=False,
    dest="composite",
    help="Make composite objects (1 SEG and 1 SR that contain all segmentations/measurement for all nodes/annotations). Composite objects will not be generated by default."
  )
  parser.add_argument(
    '--skip',
    action="store_true",
    default=False,
    dest="skip",
    help="Do not encode empty slices in the DICOM SEG objects. Empty slices will not be skipped by default."
  )
  parser.add_argument(
    '--images-dir',
    dest="imagesDir",
    help="Directory with the CT images of the LIDC-IDRI collection. The directory should be organized following this pattern: <subject ID>/<study UID>/<series UID>."
  )

  args = parser.parse_args()

  if args.logFile:
    root = logging.getLogger()
    logging.basicConfig(filename=args.logFile,level=logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)s: %(levelname)s: %(message)s')
    handler.setFormatter(formatter)
    root.addHandler(handler)
  else:
    logging.basicConfig(level=logging.INFO)
  logger = logging.getLogger("lidc2dicom")

  converter = LIDC2DICOMConverter(args)

  if args.outputDir:
    converter.tempDir = args.outputDir

  if args.subjectIDs:
    logger.info("Processing subjects "+str(args.subjectIDs))
    for s in args.subjectIDs:
      converter.convertForSubject(s)
      if args.composite:
        converter.makeCompositeObjects(s)
  elif args.subjectRange is not None and len(args.subjectRange):
    logger.info("Processing subjects from "+str(args.subjectRange[0])+" to "+str(args.subjectRange[1])+" inclusive")
    if args.subjectRange[1]<args.subjectRange[0]:
      logger.error("Invalid range.")
    for s in range(args.subjectRange[0],args.subjectRange[1]+1,1):
      converter.convertForSubject(s)
      if args.composite:
        converter.makeCompositeObjects(s)
  elif args.allSubjects:
    logging.info("Processing all subjects from 1 to 1012.")
    for s in range(1,1013,1):
      converter.convertForSubject(s)
      if args.composite:
        converter.makeCompositeObjects(s)

if __name__ == "__main__":
  main()
