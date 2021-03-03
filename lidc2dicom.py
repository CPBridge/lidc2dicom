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
        self.output_dir= args.output_dir

        self.sr_template = "sr_conversion_template.json"
        self.colors_file = "GenericColors.txt"

        # read GenericColors
        self.colors = []
        with open(self.colors_file,'r') as f:
            for l in f:
                if l.startswith('#'):
                    continue
                self.colors.append([int(c) for c in l.split(' ')[2:5]])

        self.concepts_dictionary = {}
        self.values_dictionary = {}
        with open("concepts_dict.json") as cf:
            self.concepts_dictionary = json.load(cf)
        with open("values_dict.json") as vf:
            self.values_dictionary = json.load(vf)

    def clean_up_temp_dir(self, dir):
        for p in Path(dir).glob("*.nrrd"):
            p.unlink()

    def convert_single_annotation(self, n_count, a_count, a, ct_datasets, nodule_uid, series_dir, scan):

        # update as necessary!
        nodule_name = f"Nodule {n_count + 1}"
        seg_name = f"Nodule {n_count + 1} - Annotation {a._nodule_id}"

        # Identify pylidc as the "algorithm" creating the annotations
        pylidc_algo_id = AlgorithmIdentification(name='pylidc', version=pl.__version__)

        seg_desc = SegmentDescription(
            segment_number=1,
            segment_label=seg_name,
            segmented_property_category=codes.SCT.MorphologicallyAbnormalStructure,
            segmented_property_type=codes.SCT.Nodule,
            algorithm_type=SegmentAlgorithmTypeValues.MANUAL,
            algorithm_identification=None,
            tracking_uid=nodule_uid,
            tracking_id=nodule_name,
            anatomic_regions=[codes.SCT.Lung],
        )
        seg_desc.SegmentDescription = seg_name
        seg_desc.RecommendedDisplayCIELabValue = self.colors[a_count + 1]

        self.instance_count = self.instance_count + 1
        if ct_datasets[0].SeriesNumber != '':
            series_num = int(ct_datasets[0].SeriesNumber) + self.instance_count
        else:
            series_num = self.instance_count

        dcm_seg_file = os.path.join(self.temp_subject_dir, seg_name + '.dcm')

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
            series_description=f"Segmentation of {seg_name}"
        )

        # Add in some extra information
        seg_dcm.BodyPartExamined = "Lung"
        seg_dcm.ClinicalTrialSeriesID = "Session1"
        seg_dcm.ClinicalTrialTimePointID = "1"
        seg_dcm.ClinicalTrialCoordinatingCenterName = "TCIA"
        seg_dcm.ContentLabel = "SEGMENTATION"

        # Save the file
        seg_dcm.save_as(dcm_seg_file)

        seg_uid = None
        try:
            seg_dcm = pydicom.read_file(dcm_seg_file)
            seg_uid = seg_dcm.SOPInstanceUID
        except:
            self.logger.error("Failed to read Segmentation file")
            return

        sr_name = seg_name + " evaluations"

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

        self.instance_count = self.instance_count + 1
        if ct_datasets[0].SeriesNumber != '':
            series_number = str(int(ct_datasets[0].SeriesNumber) + self.instance_count)
        else:
            series_number = str(self.instance_count)

        qualitative_evaluations = []
        for attribute in self.concepts_dictionary.keys():
            try:
                qualitative_evaluations.append(
                    CodeContentItem(
                        name=CodedConcept(**self.concepts_dictionary[attribute]),
                        value=CodedConcept(**self.values_dictionary[attribute][str(getattr(a, attribute))])
                    )
                )
            except KeyError:
                self.logger.info(f"Skipping invalid attribute: {attribute} {getattr(a, attribute)}")
                continue

        sr_name = f"Nodule {n_count + 1} - Annotation {a._nodule_id} measurements"

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
                    uid=nodule_uid,
                    identifier=nodule_name
                ),
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
            is_complete=True,
            is_verified=True,
            verifying_observer_name='anonymous',
            verifying_organization='anonymous',
            series_description=sr_name
        )

        dcm_sr_file = os.path.join(self.temp_subject_dir, sr_name + '.dcm')
        sr_dcm.save_as(dcm_sr_file)

        if not os.path.exists(dcm_sr_file):
            self.logger.error("Failed to access output SR file for " + s)


    def convert_for_subject(self, subjectID):
        s = 'LIDC-IDRI-%04i' % subjectID
        self.logger.info("Processing subject %s" % (s))
        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == s)
        self.logger.info(" Found %d scans" % (scans.count()))

        for scan in scans:
            study_uid = scan.study_instance_uid
            series_uid = scan.series_instance_uid
            series_dir = Path(scan.get_path_to_dicom_files())
            if not os.path.exists(series_dir):
                self.logger.error("Files not found for subject " + s)
                return

            try:
                ct_datasets = scan.load_all_dicom_images()
            except:
                logger.error("Failed to read input CT files")
                return

            ok = lidc_helpers.checkSeriesGeometry(str(series_dir))
            if not ok:
                self.logger.warning("Geometry inconsistent for subject %s" % (s))

            self.temp_subject_dir = os.path.join(self.output_dir, s, study_uid, series_uid)
            os.makedirs(self.temp_subject_dir, exist_ok=True)

            # now iterate over all nodules available for this subject
            anns = scan.annotations
            self.logger.info("Have %d annotations for subject %s" % (len(anns), s))

            self.instance_count = 0

            clustered_annotation_ids = []

            for n_count, nodule in enumerate(scan.cluster_annotations()):
                nodule_uid = pydicom.uid.generate_uid(prefix=None) # by default, pydicom uses 2.25 root

                for a_count, a in enumerate(nodule):
                    clustered_annotation_ids.append(a.id)
                    self.convert_single_annotation(n_count, a_count, a, ct_datasets, nodule_uid, series_dir, scan)

            if len(clustered_annotation_ids) != len(anns):
                self.logger.warning("%d annotations unaccounted for!" % (len(anns) - len(clustered_annotation_ids)))

            for ua in anns:
                if ua.id not in clustered_annotation_ids:
                    a_count = a_count + 1
                    n_count = n_count + 1
                    nodule_uid = pydicom.uid.generate_uid(prefix=None)
                    self.convert_single_annotation(n_count, a_count, ua, ct_datasets, nodule_uid, series_dir, scan)

    def make_composite_objects(self, subjectID):

        # convert all segmentations and measurements into composite objects
        # 1. find all segmentations
        # 2. read all, append metadata
        # 3. find all measurements
        # 4. read all, append metadata
        import re
        s = 'LIDC-IDRI-%04i' % subjectID
        self.logger.info("Making composite objects for " + s)

        scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == s)
        self.logger.info(" Found %d scans" % (scans.count()))

        # cannot just take all segmentation files in a folder, since

        for scan in scans:
            study_uid = scan.study_instance_uid
            series_uid = scan.series_instance_uid
            series_dir = scan.get_path_to_dicom_files()
            if not os.path.exists(series_dir):
                self.logger.error("Files not found for subject " + s)
                return

            dcm_files = glob.glob(os.path.join(series_dir, "*.dcm"))
            if not len(dcm_files):
                logger.error("No DICOM files found for subject " + s)
                return

            first_file = os.path.join(series_dir, dcm_files[0])

            try:
                ct_dcm = pydicom.read_file(first_file)
            except:
                logger.error("Failed to read input file " + first_file)
                return

            self.instance_count = 1000

            subject_scan_temp_dir = os.path.join(self.output_dir, s, study_uid, series_uid)
            all_segmentations = glob.glob(os.path.join(subject_scan_temp_dir, 'Nodule*Annotation*.nrrd'))

            if not len(all_segmentations):
                continue

            segMetadata = {}
            nrrdSegFileList = ""
            srMetadata = {}

            for segID, seg in enumerate(all_segmentations):

                prefix = seg[:-5]
                matches = re.match('Nodule (\d+) - Annotation (.+)\.', os.path.split(seg)[1])
                print("Nodule: " + matches.group(1)+" Annotation: " + matches.group(2))

                if not segMetadata:
                    segMetadata = json.load(open(prefix+".json"))
                else:
                    thisSegMetadata = json.load(open(prefix+".json"))
                    segMetadata["segmentAttributes"].append(thisSegMetadata["segmentAttributes"][0])

                if not srMetadata:
                    srMetadata = json.load(open(prefix+" measurements.json"))
                else:
                    thisSRMetadata = json.load(open(prefix+" measurements.json"))
                    thisSRMetadata["Measurements"][0]["ReferencedSegment"] = segID + 1
                    srMetadata["Measurements"].append(thisSRMetadata["Measurements"][0])

                nrrdSegFileList = nrrdSegFileList + seg + ","

            segMetadata["ContentDescription"] = "Lung nodule segmentation - all"
            segMetadata["SeriesDescription"] = "Segmentations of all nodules"
            segMetadata["SeriesNumber"] = str(int(ct_dcm.SeriesNumber) + self.instance_count)
            self.instance_count = self.instance_count + 1

            # run SEG converter

            allSegsJSON = os.path.join(subject_scan_temp_dir, "all_segmentations.json")
            with open(allSegsJSON,"w") as f:
                json.dump(segMetadata, f, indent=2)

            compositeSEGFileName = os.path.join(subject_scan_temp_dir,"all_segmentations.dcm")
            nrrdSegFileList = nrrdSegFileList[:-1]

            converterCmd = ['itkimage2segimage', "--inputImageList", nrrdSegFileList, "--inputDICOMDirectory", series_dir, "--inputMetadata", allSegsJSON, "--outputDICOM", compositeSEGFileName]
            if self.args.skip:
                converterCmd.append('--skip')
            self.logger.info("Converting to DICOM SEG with " + str(converterCmd))

            sp = subprocess.Popen(converterCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (stdout, stderr) = sp.communicate()
            self.logger.info("itkimage2segimage stdout: " + stdout.decode('ascii'))
            self.logger.warning("itkimage2segimage stderr: " + stderr.decode('ascii'))

            if not os.path.exists(compositeSEGFileName):
                self.logger.error("Failed to access output composite SEG file for " + s)

            # populate composite SR JSON
            # need SEG SOPInstnaceUID for that purpose
            segDcm = pydicom.read_file(compositeSEGFileName)
            seg_uid = segDcm.SOPInstanceUID
            ctSeriesUID = segDcm.ReferencedSeriesSequence[0].SeriesInstanceUID

            for mItem in range(len(srMetadata["Measurements"])):
                srMetadata["Measurements"][mItem]["segmentationSOPInstanceUID"] = seg_uid

            srMetadata["compositeContext"] = [os.path.split(compositeSEGFileName)[1]]

            srMetadata["ContentDescription"] = "Lung nodule measurements - all"
            srMetadata["SeriesDescription"] = "Evaluations for all nodules"
            srMetadata["SeriesNumber"] = str(int(ct_dcm.SeriesNumber) + self.instance_count)
            self.instance_count = self.instance_count + 1

            allSrsJSON = os.path.join(subject_scan_temp_dir, "all_measurements.json")
            with open(allSrsJSON,"w") as f:
                json.dump(srMetadata, f, indent=2)

            compositeSRFileName = os.path.join(subject_scan_temp_dir,"all_measurements.dcm")
            nrrdSegFileList = nrrdSegFileList[:-1]

            converterCmd = ['tid1500writer', "--inputMetadata", allSrsJSON, "--inputImageLibraryDirectory", series_dir, "--inputCompositeContextDirectory", subject_scan_temp_dir, "--outputDICOM", compositeSRFileName]
            self.logger.info("Converting to DICOM SR with " + str(converterCmd))

            sp = subprocess.Popen(converterCmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (stdout, stderr) = sp.communicate()
            self.logger.info("tid1500writer stdout: " + stdout.decode('ascii'))
            self.logger.warning("tid1500writer stderr: " + stderr.decode('ascii'))

            if not os.path.exists(compositeSRFileName):
                self.logger.error("Failed to access output composite SR file for " + s)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        usage="%(prog)s --subjects <LIDC_subjectID>\n\n"
        "This program will parse the DICOM and XML data for LIDC subject specified and generate"
        "DICOM representation for the segmentations and evaluations of the segmented nodule."
        "More details in a document to follow"
    )
    parser.add_argument(
        '--subject-range',
        dest="subject_range",
        nargs=2,
        type=int,
        help="Range of subject identifiers to be processed. Overrides individual subjects specified."
    )
    parser.add_argument(
        '--all-subjects',
        dest="all_subjects",
        action="store_true",
        help="Process all subjects (up to 1012). Overrides all other subject specifications."
    )
    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        dest="subject_ids",
        help='Identifier(s) (separated by space) of the subject to be processed.'
    )
    parser.add_argument(
        '--log',
        dest="log_file",
        help="Location of the file to store processing log."
    )
    parser.add_argument(
        '--output-dir',
        dest="output_dir",
        help="Directory for storing the results of conversion."
    )
    parser.add_argument(
        '--composite',
        action="store_true",
        default=False,
        dest="composite",
        help="Make composite objects (1 SEG and 1 SR that contain all segmentations/measurement for "
              "all nodes/annotations). Composite objects will not be generated by default."
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
        dest="images_dir",
        help="Directory with the CT images of the LIDC-IDRI collection. The directory should be organized "
             "following this pattern: <subject ID>/<study UID>/<series UID>."
    )

    args = parser.parse_args()

    if args.log_file:
        root = logging.getLogger()
        logging.basicConfig(filename=args.log_file, level=logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(name)s: %(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        root.addHandler(handler)
    else:
        logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("lidc2dicom")

    converter = LIDC2DICOMConverter(args)

    if args.output_dir:
        converter.output_dir = args.output_dir

    if args.subject_ids:
        logger.info(f"Processing subjects {args.subject_ids}")
        for s in args.subject_ids:
            converter.convert_for_subject(s)
            if args.composite:
                converter.make_composite_objects(s)
    elif args.subject_range is not None and len(args.subject_range):
        logger.info(f"Processing subjects from {args.subject_range[0]} to {args.subject_range[1]} inclusive")
        if args.subject_range[1] < args.subject_range[0]:
            logger.error("Invalid range.")
        for s in range(args.subject_range[0], args.subject_range[1] + 1, 1):
            converter.convert_for_subject(s)
            if args.composite:
                converter.make_composite_objects(s)
    elif args.all_subjects:
        logging.info("Processing all subjects from 1 to 1012.")
        for s in range(1, 1013, 1):
            converter.convert_for_subject(s)
            if args.composite:
                converter.make_composite_objects(s)

if __name__ == "__main__":
    main()
