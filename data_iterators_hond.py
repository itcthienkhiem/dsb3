import numpy as np
import utils_lung
import pathfinder
import utils


class LunaDataGenerator(object):
    def __init__(self, data_path, transform_params, data_prep_fun, rng,
                 random, infinite, patient_ids=None, **kwargs):

        if patient_ids:
            self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids]
        else:
            patient_paths = utils_lung.get_patient_data_paths(data_path)
            self.patient_paths = [p for p in patient_paths if '.mhd' in p]

        self.id2annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs)):
                idx = rand_idxs[pos]

                patient_path = self.patient_paths[idx]
                pid = utils_lung.extract_pid_filename(patient_path)

                img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                x, y, annotations, tf_matrix = self.data_prep_fun(data=img,
                                                                  pixel_spacing=pixel_spacing,
                                                                  luna_annotations=
                                                                  self.id2annotations[pid],
                                                                  luna_origin=origin)

                x = np.float32(x)[None, None, :, :, :]
                y = np.float32(y)[None, None, :, :, :]

                yield x, y, None, annotations, tf_matrix, pid

            if not self.infinite:
                break


class LunaScanPositiveDataGenerator(LunaDataGenerator):
    def __init__(self, data_path, transform_params, data_prep_fun, rng,
                 random, infinite, patient_ids=None, **kwargs):
        super(LunaScanPositiveDataGenerator, self).__init__(data_path, transform_params, data_prep_fun, rng,
                                                            random, infinite, patient_ids, **kwargs)
        patient_ids_all = [utils_lung.extract_pid_filename(p) for p in self.patient_paths]
        patient_ids_pos = [pid for pid in patient_ids_all if pid in self.id2annotations.keys()]
        self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids_pos]
        self.nsamples = len(self.patient_paths)


class LunaScanPositiveLungMaskDataGenerator(LunaScanPositiveDataGenerator):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):
        super(LunaScanPositiveLungMaskDataGenerator, self).__init__(data_path, transform_params,
                                                                    data_prep_fun, rng,
                                                                    random, infinite, patient_ids, **kwargs)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs)):
                idx = rand_idxs[pos]

                patient_path = self.patient_paths[idx]
                pid = utils_lung.extract_pid_filename(patient_path)

                img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                x, y, lung_mask, annotations, tf_matrix = self.data_prep_fun(data=img,
                                                                             pixel_spacing=pixel_spacing,
                                                                             luna_annotations=
                                                                             self.id2annotations[pid],
                                                                             luna_origin=origin)

                x = np.float32(x)[None, None, :, :, :]
                y = np.float32(y)[None, None, :, :, :]
                lung_mask = np.float32(lung_mask)[None, None, :, :, :]

                yield x, y, lung_mask, annotations, tf_matrix, pid

            if not self.infinite:
                break


class LunaScanPositiveLungMaskDataGenerator2(LunaScanPositiveDataGenerator):
    def __init__(self, data_path, lung_masks_path, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):
        super(LunaScanPositiveLungMaskDataGenerator2, self).__init__(data_path, transform_params,
                                                                     data_prep_fun, rng,
                                                                     random, infinite, patient_ids, **kwargs)

        self.lung_masks_path = lung_masks_path

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs)):
                idx = rand_idxs[pos]

                patient_path = self.patient_paths[idx]
                pid = utils_lung.extract_pid_filename(patient_path)

                lung_mask_in, origin_mask, pixel_spacing_mask = utils_lung.read_mhd(
                    self.lung_masks_path + '/%s.mhd' % pid)
                img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                x, y, lung_mask, annotations, tf_matrix = self.data_prep_fun(data=img,
                                                                             lung_mask=lung_mask_in,
                                                                             pixel_spacing=pixel_spacing,
                                                                             luna_annotations=
                                                                             self.id2annotations[pid],
                                                                             luna_origin=origin)

                x = np.float32(x)[None, None, :, :, :]
                y = np.float32(y)[None, None, :, :, :]
                lung_mask = np.float32(lung_mask)[None, None, :, :, :]

                yield x, y, lung_mask, annotations, tf_matrix, pid

            if not self.infinite:
                break


class PatchPositiveLunaDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, data_prep_fun, rng,
                 full_batch, random, infinite, patient_ids=None, **kwargs):

        self.id2annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

        if patient_ids:
            self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids]
        else:
            patient_paths = utils_lung.get_patient_data_paths(data_path)
            self.patient_paths = [p for p in patient_paths if '.mhd' in p]

        patient_ids_all = [utils_lung.extract_pid_filename(p) for p in self.patient_paths]
        patient_ids_pos = [pid for pid in patient_ids_all if pid in self.id2annotations.keys()]
        self.patient_paths = [data_path + '/' + p + '.mhd' for p in patient_ids_pos]

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.batch_size = batch_size
        self.full_batch = full_batch

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    id = utils_lung.extract_pid_filename(patient_path)
                    patients_ids.append(id)
                    img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)

                    patient_annotations = self.id2annotations[id]
                    patch_center = patient_annotations[self.rng.randint(len(patient_annotations))]
                    x_batch[i, 0, :, :, :], y_batch[i, 0, :, :, :] = self.data_prep_fun(data=img,
                                                                                        patch_center=patch_center,
                                                                                        pixel_spacing=pixel_spacing,
                                                                                        luna_annotations=patient_annotations,
                                                                                        luna_origin=origin)
                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:
                break


class ValidPatchPositiveLunaDataGenerator(object):
    def __init__(self, data_path, transform_params, patient_ids, data_prep_fun, **kwargs):

        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

        self.id2positive_annotations = {}
        self.id2patient_path = {}
        n_positive = 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                n_pos = len(id2positive_annotations[pid])
                self.id2patient_path[pid] = data_path + '/' + pid + '.mhd'
                n_positive += n_pos

        self.nsamples = n_positive
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):

        for pid in self.id2positive_annotations.iterkeys():
            for patch_center in self.id2positive_annotations[pid]:
                patient_path = self.id2patient_path[pid]
                img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)

                patient_annotations = self.id2positive_annotations[pid]
                x_batch, y_batch = self.data_prep_fun(data=img,
                                                      patch_center=patch_center,
                                                      pixel_spacing=pixel_spacing,
                                                      luna_annotations=patient_annotations,
                                                      luna_origin=origin)
                x_batch = np.float32(x_batch)[None, None, :, :, :]
                y_batch = np.float32(y_batch)[None, None, :, :, :]
                yield x_batch, y_batch, [pid]


class CandidatesLunaDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, patient_ids, data_prep_fun, rng,
                 full_batch, random, infinite, positive_proportion, nodule_size_bins=None, **kwargs):

        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.patient_paths = []
        n_positive, n_negative = 0, 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                self.id2negative_annotations[pid] = id2negative_annotations[pid]
                self.patient_paths.append(data_path + '/' + pid + self.file_extension)
                n_positive += len(id2positive_annotations[pid])
                n_negative += len(id2negative_annotations[pid])

        print 'n positive', n_positive
        print 'n negative', n_negative

        self.nsamples = len(self.patient_paths)

        print 'n patients', self.nsamples
        self.data_path = data_path
        self.batch_size = batch_size
        self.rng = rng
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.positive_proportion = positive_proportion
        self.nodule_size_bins = nodule_size_bins
        if nodule_size_bins is not None:
            assert self.transform_params['pixel_spacing'] == (1., 1., 1.)

    def find_nodule_size_bin(self, diameter):
        if diameter == 0:
            return 0
        else:
            return np.digitize(diameter, self.nodule_size_bins)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]

                    id = utils_lung.extract_pid_filename(patient_path, self.file_extension)
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                        if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
                    if i < np.rint(self.batch_size * self.positive_proportion):
                        patient_annotations = self.id2positive_annotations[id]
                    else:
                        patient_annotations = self.id2negative_annotations[id]

                    patch_center = patient_annotations[self.rng.randint(len(patient_annotations))]

                    y_batch[i] = float(patch_center[-1] > 0) if self.nodule_size_bins is None else \
                        self.find_nodule_size_bin(patch_center[-1])
                    x_batch[i, 0, :, :, :] = self.data_prep_fun(data=img,
                                                                patch_center=patch_center,
                                                                pixel_spacing=pixel_spacing,
                                                                luna_origin=origin)

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:
                break


# <<<<<<< HEAD
# class CandidatesLunaDataGeneratorBetter(object):
#     def __init__(self, data_path, batch_size, transform_params, patient_ids, data_prep_fun, rng,
#                  full_batch, random, infinite, positive_proportion,
#                  label_prep_fun=None, random_negative_samples=False, **kwargs):
# =======
class CandidatesPropertiesLunaDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, label_prep_fun,
                 nproperties,  patient_ids, data_prep_fun, rng,
                 full_batch, random, infinite, positive_proportion, properties_included=[],
                 random_negative_samples=False, **kwargs):

        id2positive_annotations = utils_lung.read_luna_properties(pathfinder.LUNA_PROPERTIES_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.pid2patient_path = {}
        n_positive = 0
        for pid in patient_ids:
            self.pid2patient_path[pid] = data_path + '/' + pid + self.file_extension
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                n_positive += len(id2positive_annotations[pid])
            if pid in id2negative_annotations:
                self.id2negative_annotations[pid] = id2negative_annotations[pid]

        self.nsamples = int(n_positive + (1. - positive_proportion) / positive_proportion * n_positive)
        print 'n samples', self.nsamples
        self.idx2pid_annotation = {}
        i = 0
        for pid, annotations in self.id2positive_annotations.iteritems():
            for a in annotations:
                self.idx2pid_annotation[i] = (pid, a)
                i += 1
        print 'n positive', len(self.idx2pid_annotation.keys())

        if random_negative_samples:
            while i < self.nsamples:
                self.idx2pid_annotation[i] = (None, None)
                i += 1
        else:
            while i < self.nsamples:
                pid = rng.choice(self.id2negative_annotations.keys())
                patient_annotations = self.id2negative_annotations[pid]
                a = patient_annotations[rng.randint(len(patient_annotations))]
                self.idx2pid_annotation[i] = (pid, a)
                i += 1
        assert len(self.idx2pid_annotation) == self.nsamples

        self.data_path = data_path
        self.batch_size = batch_size
        self.rng = rng
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.positive_proportion = positive_proportion
        self.label_prep_fun = label_prep_fun
        self.nlabels = nproperties

        if len(properties_included)>0:
            self.nlabels=len(properties_included)
        self.properties_included = properties_included

        assert self.transform_params['pixel_spacing'] == (1., 1., 1.)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((nb, self.nlabels), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    pid, patch_annotation = self.idx2pid_annotation[idx]

                    if pid is None:
                        pid = self.rng.choice(self.id2negative_annotations.keys())
                        patient_annotations = self.id2negative_annotations[pid]
                        patch_annotation = patient_annotations[self.rng.randint(len(patient_annotations))]

                    patient_path = self.pid2patient_path[pid]
                    patients_ids.append(pid)

                    y_batch[i] = self.label_prep_fun(patch_annotation,self.properties_included)
                    # print pid, y_batch[i]

                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                        if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)

                    patch_zyxd = patch_annotation[:4]
                    x_batch[i, 0, :, :, :] = self.data_prep_fun(data=img,
                                                                patch_center=patch_zyxd,
                                                                pixel_spacing=pixel_spacing,
                                                                luna_origin=origin)
                y_batch = np.asarray(y_batch,dtype=np.float32)
                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:
                break


class CandidatesLunaValidDataGenerator(object):
    def __init__(self, data_path, transform_params, patient_ids, data_prep_fun, label_prep_fun=None,properties_included=[],
                 **kwargs):
        rng = np.random.RandomState(42)  # do not change this!!!
#>>>>>>> dsb-ira-frederic2

        id2positive_annotations = utils_lung.read_luna_properties(pathfinder.LUNA_PROPERTIES_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.pid2patient_path = {}
        n_positive = 0
        for pid in patient_ids:
            self.pid2patient_path[pid] = data_path + '/' + pid + self.file_extension
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                n_positive += len(id2positive_annotations[pid])
            if pid in id2negative_annotations:
                self.id2negative_annotations[pid] = id2negative_annotations[pid]

        self.nsamples = int(n_positive + (1. - positive_proportion) / positive_proportion * n_positive)
        print 'n samples', self.nsamples
        self.idx2pid_annotation = {}
        i = 0
        for pid, annotations in self.id2positive_annotations.iteritems():
            for a in annotations:
                self.idx2pid_annotation[i] = (pid, a)
                i += 1
        print 'n positive', len(self.idx2pid_annotation.keys())

        if random_negative_samples:
            while i < self.nsamples:
                self.idx2pid_annotation[i] = (None, None)
                i += 1
        else:
            while i < self.nsamples:
                pid = rng.choice(self.id2negative_annotations.keys())
                patient_annotations = self.id2negative_annotations[pid]
                a = patient_annotations[rng.randint(len(patient_annotations))]
                self.idx2pid_annotation[i] = (pid, a)
                i += 1
        assert len(self.idx2pid_annotation) == self.nsamples

        self.data_path = data_path
        self.batch_size = batch_size
        self.rng = rng
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.positive_proportion = positive_proportion
        self.label_prep_fun = label_prep_fun
        if label_prep_fun is not None:
            assert self.transform_params['pixel_spacing'] == (1., 1., 1.)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    pid, patch_center = self.idx2pid_annotation[idx]

                    if pid is None:
                        pid = self.rng.choice(self.id2negative_annotations.keys())
                        patient_annotations = self.id2negative_annotations[pid]
                        patch_center = patient_annotations[self.rng.randint(len(patient_annotations))]

                    patient_path = self.pid2patient_path[pid]
                    patients_ids.append(pid)

                    y_batch[i] = float(patch_center[-1] > 0) if self.label_prep_fun is None else \
                        self.label_prep_fun(patch_center[-1])

                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                        if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)

                    x_batch[i, 0, :, :, :] = self.data_prep_fun(data=img,
                                                                patch_center=patch_center,
                                                                pixel_spacing=pixel_spacing,
                                                                luna_origin=origin)

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:
                break


class CandidatesPropertiesLunaDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, label_prep_fun,
                 nproperties, patient_ids, data_prep_fun, rng,
                 full_batch, random, infinite, positive_proportion,
                 random_negative_samples=False, **kwargs):

        id2positive_annotations = utils_lung.read_luna_properties(pathfinder.LUNA_PROPERTIES_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.pid2patient_path = {}
        n_positive = 0
        for pid in patient_ids:
            self.pid2patient_path[pid] = data_path + '/' + pid + self.file_extension
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                n_positive += len(id2positive_annotations[pid])
            if pid in id2negative_annotations:
                self.id2negative_annotations[pid] = id2negative_annotations[pid]

        self.nsamples = int(n_positive + (1. - positive_proportion) / positive_proportion * n_positive)
        print 'n samples', self.nsamples
        self.idx2pid_annotation = {}
        i = 0
        for pid, annotations in self.id2positive_annotations.iteritems():
            for a in annotations:
                self.idx2pid_annotation[i] = (pid, a)
                i += 1
        print 'n positive', len(self.idx2pid_annotation.keys())

        if random_negative_samples:
            while i < self.nsamples:
                self.idx2pid_annotation[i] = (None, None)
                i += 1
        else:
            while i < self.nsamples:
                pid = rng.choice(self.id2negative_annotations.keys())
                patient_annotations = self.id2negative_annotations[pid]
                a = patient_annotations[rng.randint(len(patient_annotations))]
                self.idx2pid_annotation[i] = (pid, a)
                i += 1
        assert len(self.idx2pid_annotation) == self.nsamples

        self.data_path = data_path
        self.batch_size = batch_size
        self.rng = rng
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.positive_proportion = positive_proportion
        self.label_prep_fun = label_prep_fun
        self.nlabels = nproperties
        assert self.transform_params['pixel_spacing'] == (1., 1., 1.)

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((nb, self.nlabels), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    pid, patch_annotation = self.idx2pid_annotation[idx]

                    if pid is None:
                        pid = self.rng.choice(self.id2negative_annotations.keys())
                        patient_annotations = self.id2negative_annotations[pid]
                        patch_annotation = patient_annotations[self.rng.randint(len(patient_annotations))]

                    patient_path = self.pid2patient_path[pid]
                    patients_ids.append(pid)

                    y_batch[i] = self.label_prep_fun(patch_annotation)

                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                        if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)

                    patch_zyxd = patch_annotation[:4]
                    x_batch[i, 0, :, :, :] = self.data_prep_fun(data=img,
                                                                patch_center=patch_zyxd,
                                                                pixel_spacing=pixel_spacing,
                                                                luna_origin=origin)

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:
                break


class CandidatesPropertiesLunaValidDataGenerator(object):
    def __init__(self, data_path, transform_params, patient_ids, data_prep_fun, label_prep_fun,
                 nproperties, **kwargs):
        rng = np.random.RandomState(42)  # do not change this!!!

        id2positive_annotations = utils_lung.read_luna_properties(pathfinder.LUNA_PROPERTIES_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.id2patient_path = {}
        n_positive, n_negative = 0, 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                negative_annotations = id2negative_annotations[pid]
                n_pos = len(id2positive_annotations[pid])
                n_neg = len(id2negative_annotations[pid])
                neg_idxs = rng.choice(n_neg, size=n_pos, replace=False)
                negative_annotations_selected = []
                for i in neg_idxs:
                    negative_annotations_selected.append(negative_annotations[i])
                self.id2negative_annotations[pid] = negative_annotations_selected

                self.id2patient_path[pid] = data_path + '/' + pid + self.file_extension
                n_positive += n_pos
                n_negative += n_pos

        print 'n positive', n_positive
        print 'n negative', n_negative

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.rng = rng
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.label_prep_fun = label_prep_fun
# <<<<<<< HEAD
#         self.nlabels = nproperties
#         assert self.transform_params['pixel_spacing'] == (1., 1., 1.)
# =======
        if label_prep_fun is not None:
            assert self.transform_params['pixel_spacing'] == (1., 1., 1.)

        self.properties_included = properties_included
#>>>>>>> dsb-ira-frederic2

    def generate(self):

        for pid in self.id2positive_annotations.iterkeys():
            for patch_annotation in self.id2positive_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                    if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
# <<<<<<< HEAD
#                 y_batch = np.array(self.label_prep_fun(patch_annotation), dtype='float32')[None, :]
#                 x_batch = np.float32(self.data_prep_fun(data=img,
#                                                         patch_center=patch_annotation[:4],
# =======
                if self.label_prep_fun is None:
                    y_batch = np.array([[1.]], dtype='float32')
                else:
                    y_batch = np.array([self.label_prep_fun(patch_center,self.properties_included)], dtype='float32')
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center[0:4],
#>>>>>>> dsb-ira-frederic2
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]

            for patch_annotation in self.id2negative_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                    if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
                y_batch = np.array(self.label_prep_fun(patch_annotation), dtype='float32')[None, :]
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_annotation[:4],
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]


class CandidatesMTLunaDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, patient_ids, data_prep_fun, rng,
                 full_batch, random, infinite, positive_proportion, **kwargs):
        """
        :param transform_params: this has to be a list with transformations
        """
        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.patient_paths = []
        n_positive, n_negative = 0, 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                self.id2negative_annotations[pid] = id2negative_annotations[pid]
                self.patient_paths.append(data_path + '/' + pid + self.file_extension)
                n_positive += len(id2positive_annotations[pid])
                n_negative += len(id2negative_annotations[pid])

        print 'n positive', n_positive
        print 'n negative', n_negative

        self.nsamples = len(self.patient_paths)

        print 'n patients', self.nsamples
        self.data_path = data_path
        self.batch_size = batch_size
        self.rng = rng
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        assert isinstance(self.transform_params, list)
        self.positive_proportion = positive_proportion

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                y_batch = np.zeros((nb, 1), dtype='float32')
                xs_batch = []

                for transformation in self.transform_params:
                    x_batch = np.zeros((nb, 1) + transformation['patch_size'], dtype='float32')
                    xs_batch.append(x_batch)
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]

                    id = utils_lung.extract_pid_filename(patient_path, self.file_extension)
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                        if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
                    if i < np.rint(self.batch_size * self.positive_proportion):
                        patient_annotations = self.id2positive_annotations[id]
                    else:
                        patient_annotations = self.id2negative_annotations[id]

                    patch_center = patient_annotations[self.rng.randint(len(patient_annotations))]

                    y_batch[i] = float(patch_center[-1] > 0)
                    x_tf = self.data_prep_fun(data=img,
                                              patch_center=patch_center,
                                              pixel_spacing=pixel_spacing,
                                              luna_origin=origin)
                    for k in xrange(len(x_tf)):
                        xs_batch[k][i, 0, :, :, :] = x_tf[k]

                if self.full_batch:
                    if nb == self.batch_size:
                        yield xs_batch, y_batch, patients_ids
                else:
                    yield xs_batch, y_batch, patients_ids

            if not self.infinite:
                break


class CandidatesMTValidLunaDataGenerator(object):
    def __init__(self, data_path, transform_params, patient_ids, data_prep_fun, **kwargs):
        """
        :param transform_params: this has to be a list with transformations
        """
        rng = np.random.RandomState(42)  # do not change this!!!

        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.id2patient_path = {}
        n_positive, n_negative = 0, 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                negative_annotations = id2negative_annotations[pid]
                n_pos = len(id2positive_annotations[pid])
                n_neg = len(id2negative_annotations[pid])
                neg_idxs = rng.choice(n_neg, size=n_pos, replace=False)
                negative_annotations_selected = []
                for i in neg_idxs:
                    negative_annotations_selected.append(negative_annotations[i])
                self.id2negative_annotations[pid] = negative_annotations_selected

                self.id2patient_path[pid] = data_path + '/' + pid + self.file_extension
                n_positive += n_pos
                n_negative += n_pos

        print 'n positive', n_positive
        print 'n negative', n_negative

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.rng = rng
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        assert isinstance(self.transform_params, list)

    def generate(self):
        for pid in self.id2positive_annotations.iterkeys():
            for patch_center in self.id2positive_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                    if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
                y_batch = np.array([[1.]], dtype='float32')

                xs_batch = self.data_prep_fun(data=img, patch_center=patch_center,
                                              pixel_spacing=pixel_spacing,
                                              luna_origin=origin)
                for k in xrange(len(xs_batch)):
                    xs_batch[k] = np.float32(xs_batch[k])[None, None, :, :, :]

                yield xs_batch, y_batch, [pid]

            for patch_center in self.id2negative_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                    if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
                y_batch = np.array([[0.]], dtype='float32')
                xs_batch = self.data_prep_fun(data=img, patch_center=patch_center,
                                              pixel_spacing=pixel_spacing,
                                              luna_origin=origin)
                for k in xrange(len(xs_batch)):
                    xs_batch[k] = np.float32(xs_batch[k])[None, None, :, :, :]

                yield xs_batch, y_batch, [pid]


class CandidatesLunaValidDataGenerator(object):
    def __init__(self, data_path, transform_params, patient_ids, data_prep_fun, label_prep_fun=None,
                 **kwargs):
        rng = np.random.RandomState(42)  # do not change this!!!

        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.id2patient_path = {}
        n_positive, n_negative = 0, 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                negative_annotations = id2negative_annotations[pid]
                n_pos = len(id2positive_annotations[pid])
                n_neg = len(id2negative_annotations[pid])
                neg_idxs = rng.choice(n_neg, size=n_pos, replace=False)
                negative_annotations_selected = []
                for i in neg_idxs:
                    negative_annotations_selected.append(negative_annotations[i])
                self.id2negative_annotations[pid] = negative_annotations_selected

                self.id2patient_path[pid] = data_path + '/' + pid + self.file_extension
                n_positive += n_pos
                n_negative += n_pos

        print 'n positive', n_positive
        print 'n negative', n_negative

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.rng = rng
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.label_prep_fun = label_prep_fun
        if label_prep_fun is not None:
            assert self.transform_params['pixel_spacing'] == (1., 1., 1.)

    def generate(self):

        for pid in self.id2positive_annotations.iterkeys():
            for patch_center in self.id2positive_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                    if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
                if self.label_prep_fun is None:
                    y_batch = np.array([[1.]], dtype='float32')
                else:
                    y_batch = np.array([[self.label_prep_fun(patch_center[-1])]], dtype='float32')
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]

            for patch_center in self.id2negative_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                    if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
                y_batch = np.array([[0.]], dtype='float32')
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]


class CandidatesPositiveSizesLunaValidDataGenerator(object):
    def __init__(self, data_path, transform_params, patient_ids, data_prep_fun, nodule_size_bins=None,
                 **kwargs):
        rng = np.random.RandomState(42)  # do not change this!!!

        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.id2patient_path = {}
        n_positive = 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                n_pos = len(id2positive_annotations[pid])
                self.id2patient_path[pid] = data_path + '/' + pid + self.file_extension
                n_positive += n_pos

        print 'n positive', n_positive

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.rng = rng
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.nodule_size_bins = nodule_size_bins
        if nodule_size_bins is not None:
            assert self.transform_params['pixel_spacing'] == (1., 1., 1.)

    def find_nodule_size_bin(self, diameter):
        return np.digitize(diameter, self.nodule_size_bins)

    def generate(self):

        for pid in self.id2positive_annotations.iterkeys():
            for patch_center in self.id2positive_annotations[pid]:
                patient_path = self.id2patient_path[pid]

                img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                    if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
                if self.nodule_size_bins is None:
                    y_batch = np.array([[1.]], dtype='float32')
                else:
                    y_batch = np.array([[self.find_nodule_size_bin(patch_center[-1])]], dtype='float32')
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]


class FixedCandidatesLunaDataGenerator(object):
    def __init__(self, data_path, transform_params, id2candidates_path, data_prep_fun, top_n=None):

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2candidates_path = id2candidates_path
        self.id2patient_path = {}
        for pid in id2candidates_path.keys():
            self.id2patient_path[pid] = data_path + '/' + pid + self.file_extension

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.top_n = top_n

    def generate(self):

        for pid in self.id2candidates_path.iterkeys():
            patient_path = self.id2patient_path[pid]
            print 'PATIENT', pid
            candidates = utils.load_pkl(self.id2candidates_path[pid])
            if self.top_n is not None:
                candidates = candidates[:self.top_n]
                print candidates
            print 'n blobs', len(candidates)

            img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)

            for candidate in candidates:
                y_batch = np.array(candidate, dtype='float32')
                patch_center = candidate[:3]
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing,
                                                        luna_origin=origin))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]


class DSBScanDataGenerator(object):
    def __init__(self, data_path, transform_params, data_prep_fun, **kwargs):
        self.patient_paths = utils_lung.get_patient_data_paths(data_path)
        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        for p in self.patient_paths:
            pid = utils_lung.extract_pid_dir(p)

            img, pixel_spacing = utils_lung.read_dicom_scan(p)

            x, tf_matrix = self.data_prep_fun(data=img, pixel_spacing=pixel_spacing)

            x = np.float32(x)[None, None, :, :, :]
            yield x, None, tf_matrix, pid


class DSBScanLungMaskDataGenerator(object):
    def __init__(self, data_path, transform_params, data_prep_fun, exclude_pids=None,
                 include_pids=None, part_out_of=(1, 1)):

        self.patient_paths = utils_lung.get_patient_data_paths(data_path)

        this_part = part_out_of[0]
        all_parts = part_out_of[1]
        part_lenght = int(len(self.patient_paths) / all_parts)

        if this_part == all_parts:
            self.patient_paths = self.patient_paths[part_lenght * (this_part - 1):]
        else:
            self.patient_paths = self.patient_paths[part_lenght * (this_part - 1): part_lenght * this_part]

        if exclude_pids is not None:
            for ep in exclude_pids:
                for i in xrange(len(self.patient_paths)):
                    if ep in self.patient_paths[i]:
                        self.patient_paths.pop(i)
                        break

        if include_pids is not None:
            self.patient_paths = [data_path + '/' + p for p in include_pids]

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        for p in self.patient_paths:
            pid = utils_lung.extract_pid_dir(p)

            img, pixel_spacing = utils_lung.read_dicom_scan(p)

            x, lung_mask, tf_matrix = self.data_prep_fun(data=img, pixel_spacing=pixel_spacing)

            x = np.float32(x)[None, None, :, :, :]
            lung_mask = np.float32(lung_mask)[None, None, :, :, :]
            yield x, lung_mask, tf_matrix, pid


            
            
            
class AAPMScanLungMaskDataGenerator(object):
    def __init__(self, data_path, transform_params, data_prep_fun, exclude_pids=None,
                 include_pids=None, part_out_of=(1, 1)):

        # just hand back the first series here?
        self.patient_paths = utils_lung.get_patient_data_paths_aapm(data_path)

        this_part = part_out_of[0]
        all_parts = part_out_of[1]
        part_lenght = int(len(self.patient_paths) / all_parts)

        if this_part == all_parts:
            self.patient_paths = self.patient_paths[part_lenght * (this_part - 1):]
        else:
            self.patient_paths = self.patient_paths[part_lenght * (this_part - 1): part_lenght * this_part]

        # TODO: ignored this for now
        # if exclude_pids is not None:
        #     for ep in exclude_pids:
        #         for i in xrange(len(self.patient_paths)):
        #             if ep in self.patient_paths[i]:
        #                 self.patient_paths.pop(i)
        #                 break

        # if include_pids is not None:
        #     self.patient_paths = [data_path + '/' + p for p in include_pids]

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        for p in self.patient_paths:
            pid = utils_lung.extract_pid_dir_aapm(p)

            img, pixel_spacing = utils_lung.read_dicom_scan(p)

            x, lung_mask, tf_matrix = self.data_prep_fun(data=img, pixel_spacing=pixel_spacing)

            x = np.float32(x)[None, None, :, :, :]
            lung_mask = np.float32(lung_mask)[None, None, :, :, :]
            yield x, lung_mask, tf_matrix, pid



class CandidatesAAPMDataGenerator(object):
    def __init__(self, data_path, transform_params, id2candidates_path, data_prep_fun, exclude_pids=None):
        if exclude_pids is not None:
            for p in exclude_pids:
                id2candidates_path.pop(p, None)

        self.id2candidates_path = id2candidates_path
        self.id2patient_path = {}
        for pid in id2candidates_path.keys():
            self.id2patient_path[pid] =utils_lung.get_path_to_image_from_patient(data_path,pid)#: data_path + '/' + pid +

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):

        for pid in self.id2candidates_path.iterkeys():
            patient_path = self.id2patient_path[pid]
            print pid, patient_path
            img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

            print self.id2candidates_path[pid]
            candidates = utils.load_pkl(self.id2candidates_path[pid])
            print candidates.shape

            #y_batch=read_aapm_annotations(file_path)

            for candidate in candidates:
                y_batch = np.array(candidate, dtype='float32')
                patch_center = candidate[:3]
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]

            



class DSBDataGenerator(object):
    def __init__(self, data_path, transform_params, data_prep_fun, patient_pids=None, **kwargs):
        self.patient_paths = utils_lung.get_patient_data_paths(data_path)


        self.patient_paths = [data_path + '/' + p for p in patient_pids]

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):
        for p in self.patient_paths:
            pid = utils_lung.extract_pid_dir(p)

            img, pixel_spacing = utils_lung.read_dicom_scan(p)

            x, tf_matrix = self.data_prep_fun(data=img, pixel_spacing=pixel_spacing)

            x = np.float32(x)[None, None, :, :, :]
            yield x,  pid



class CandidatesDSBDataGenerator(object):
    def __init__(self, data_path, transform_params, id2candidates_path, data_prep_fun, exclude_pids=None):
        if exclude_pids is not None:
            for p in exclude_pids:
                id2candidates_path.pop(p, None)

        self.id2candidates_path = id2candidates_path
        self.id2patient_path = {}
        for pid in id2candidates_path.keys():
            self.id2patient_path[pid] = data_path + '/' + pid

        self.nsamples = len(self.id2patient_path)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params

    def generate(self):

        for pid in self.id2candidates_path.iterkeys():
            patient_path = self.id2patient_path[pid]
            print pid, patient_path
            img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

            print self.id2candidates_path[pid]
            candidates = utils.load_pkl(self.id2candidates_path[pid])
            print candidates.shape
            for candidate in candidates:
                y_batch = np.array(candidate, dtype='float32')
                patch_center = candidate[:3]
                x_batch = np.float32(self.data_prep_fun(data=img,
                                                        patch_center=patch_center,
                                                        pixel_spacing=pixel_spacing))[None, None, :, :, :]

                yield x_batch, y_batch, [pid]


class AAPMPatientsDataGenerator(object):

    def __init__(self, data_path, batch_size, transform_params, id2candidates_path, data_prep_fun,
                 n_candidates_per_patient, rng, random, infinite, shuffle_top_n=False, patient_ids=None):


        self.id2label = utils_lung.read_aapm_labels_per_patient(pathfinder.AAPM_LABELS_PATH)
        self.id2candidates_path = id2candidates_path
        self.patient_paths = []
        if patient_ids is not None:
            for pid in patient_ids:
                self.patient_paths.append(utils_lung.get_path_to_image_from_patient(data_path,pid))
        else:
            raise ValueError('provide patient ids')

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.batch_size = batch_size
        self.transform_params = transform_params
        self.n_candidates_per_patient = n_candidates_per_patient
        self.rng = rng
        self.random = random
        self.infinite = infinite
        
        #TODO: used for debugging purposes to take a look at the ground truth
        self.ground_truth=utils_lung.read_aapm_annotations(pathfinder.AAPM_LABELS_PATH)

        self.shuffle_top_n = shuffle_top_n


    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)

            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]

                x_batch = np.zeros((self.batch_size, self.n_candidates_per_patient, 1,)
                                   + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((self.batch_size,), dtype='float32')
                pids_batch = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    pid = utils_lung.extract_pid_dir_aapm(patient_path)

                    img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

                    all_candidates = utils.load_pkl(self.id2candidates_path[pid])
                    top_candidates = all_candidates[:self.n_candidates_per_patient]
                    if self.shuffle_top_n:
                        self.rng.shuffle(top_candidates)

                    # first take a look at the image 
                    # get the voxel coords of the node to 
                    #print "ground_truth: {}".format(ground_truth)
                    #utils_plots.plot_slice_3d_3axis(img, pid, img_dir=utils.get_dir_path('analysis',pathfinder.METADATA_PATH), idx=ground_truth)



                    # can I probably just use that function to hand down the one center and set world coordinates to false? that should work I think :-)

                    import utils_plots
                    ground_truth=[idx for idx in self.ground_truth[pid][0][:-1]]
                    ground_truth[0]=img.shape[0]-ground_truth[0]
                    

                    print "ground truth: {}".format(ground_truth)
                    # TODO: candidates probably need to be converted to voxel coordinates to 
                    # be comparable ?
                    print "candidates: {}".format(top_candidates)

                    # x_plot = np.float32(self.data_prep_fun(data=img,
                    #                                            patch_centers=[ground_truth],
                    #                                            pixel_spacing=pixel_spacing))[:, None, :, :, :]

                    x_batch[i] = np.float32(self.data_prep_fun(data=img,
                                                               patch_centers=top_candidates,
                                                               pixel_spacing=pixel_spacing))[:, None, :, :, :]
                    #utils_plots.plot_slice_3d_3axis(x_batch[i][0,0,:,:,:], pid, img_dir=utils.get_dir_path('analysis',pathfinder.METADATA_PATH))
                    
                    y_batch[i] = self.id2label.get(pid)
                    pids_batch.append(pid)

                if len(idxs_batch) == self.batch_size:
                    yield x_batch, y_batch, pids_batch

            if not self.infinite:
                break





class DSBPatientsDataGenerator(object):

    def __init__(self, data_path, batch_size, transform_params, id2candidates_path, data_prep_fun,
                 n_candidates_per_patient, rng, random, infinite, shuffle_top_n=False, patient_ids=None):

        self.id2label = utils_lung.read_labels(pathfinder.LABELS_PATH)
        self.id2candidates_path = id2candidates_path
        self.patient_paths = []
        if patient_ids is not None:
            for pid in patient_ids:
                self.patient_paths.append(data_path + '/' + pid)
        else:
            raise ValueError('provide patient ids')

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.batch_size = batch_size
        self.transform_params = transform_params
        self.n_candidates_per_patient = n_candidates_per_patient
        self.rng = rng
        self.random = random
        self.infinite = infinite

        self.shuffle_top_n = shuffle_top_n


    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)

            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]

                x_batch_tmp = np.zeros((self.batch_size, self.n_candidates_per_patient,)
                                   + self.transform_params['patch_size'], dtype='float32')
                x_batch = np.zeros((self.batch_size*self.n_candidates_per_patient,)
                                   + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((self.batch_size,), dtype='float32')
                pids_batch = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    pid = utils_lung.extract_pid_dir(patient_path)

                    img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

                    all_candidates = utils.load_pkl(self.id2candidates_path[pid])
                    top_candidates = all_candidates[:self.n_candidates_per_patient]
                    if self.shuffle_top_n:
                        self.rng.shuffle(top_candidates)

                    x_batch_tmp[i] = np.float32(self.data_prep_fun(data=img,
                                                               patch_centers=top_candidates,
                                                               pixel_spacing=pixel_spacing))[:, :, :, :]
                    y_batch[i] = self.id2label.get(pid)
                    pids_batch.append(pid)

                if len(idxs_batch) == self.batch_size:
                    x_batch=x_batch_tmp.reshape((self.batch_size*self.n_candidates_per_patient,)
                                   + self.transform_params['patch_size'])
                    yield x_batch, y_batch, pids_batch

            if not self.infinite:
                break


class MixedPatientsDataGenerator(object):

    def __init__(self, data_path,aapm_data_path, batch_size, transform_params, id2candidates_path,aapm_id2candidates_path, data_prep_fun,
                 n_candidates_per_patient, rng, random, infinite, shuffle_top_n=False, patient_ids=None,aapm_patient_ids=None):


        # ok we will not have colliding ids

        # build id2label and id2candidate paths
        dsb_id2label=utils_lung.read_labels(pathfinder.LABELS_PATH) 
        aapm_id2label=utils_lung.read_aapm_labels_per_patient(pathfinder.AAPM_LABELS_PATH)
        

        self.id2label = dsb_id2label.copy()#{**dsb_id2label, **aapm_id2label}
        self.id2label.update(aapm_id2label)
        #print "id2label: {}".format(self.id2label)
        self.id2candidates_path = id2candidates_path.copy()  #{id2candidates_path,aapm_id2candidates_path}
        self.id2candidates_path.update(aapm_id2candidates_path)
        #print "id2candidates_path: {}".format(self.id2candidates_path)
        self.patient_paths = []
        if patient_ids is not None:
            for pid in patient_ids:
                self.patient_paths.append((pid,data_path + '/' + pid))
        else:
            raise ValueError('provide patient ids')

        if aapm_patient_ids is not None:
           for pid in aapm_patient_ids:
               self.patient_paths.append((pid,utils_lung.get_path_to_image_from_patient(aapm_data_path,pid)))




        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.batch_size = batch_size
        self.transform_params = transform_params
        self.n_candidates_per_patient = n_candidates_per_patient
        self.rng = rng
        self.random = random
        self.infinite = infinite

        self.shuffle_top_n = shuffle_top_n


    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)

            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]

                x_batch = np.zeros((self.batch_size, self.n_candidates_per_patient, 1,)
                                   + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((self.batch_size,), dtype='float32')
                pids_batch = []

                for i, idx in enumerate(idxs_batch):
                    (pid,patient_path) = self.patient_paths[idx]
                    #pid = utils_lung.extract_pid_dir(patient_path)

                    img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

                    all_candidates = utils.load_pkl(self.id2candidates_path[pid])
                    top_candidates = all_candidates[:self.n_candidates_per_patient]
                    if self.shuffle_top_n:
                        self.rng.shuffle(top_candidates)

                    x_batch[i] = np.float32(self.data_prep_fun(data=img,
                                                               patch_centers=top_candidates,
                                                               pixel_spacing=pixel_spacing))[:, None, :, :, :]
                    y_batch[i] = self.id2label.get(pid)
                    pids_batch.append(pid)

                if len(idxs_batch) == self.batch_size:
                    yield x_batch, y_batch, pids_batch

            if not self.infinite:
                break


class CandidatesMixedMalignantBenignGenerator(object):
    def __init__(self, data_path,aapm_data_path, batch_size, transform_params, patient_ids, aapm_patient_ids, data_prep_fun,
                 rng, full_batch, random, infinite, positive_proportion):
        
        # just the labels
        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        id2positive_annotations_aapm = utils_lung.read_aapm_annotations(pathfinder.AAPM_LABELS_PATH)
        
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        id2negative_annotations_aapm = utils_lung.read_luna_negative_candidates(pathfinder.AAPM_CANDIDATES_PATH)



        self.luna_file_extension = '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.patient_paths = []
        n_positive, n_negative = 0, 0

        for pid in patient_ids:

            if pid in id2positive_annotations:

                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                self.id2negative_annotations[pid] = id2negative_annotations[pid]
                self.patient_paths.append((pid,data_path + '/' + pid + self.luna_file_extension))
                n_positive += len(id2positive_annotations[pid])
                n_negative += len(id2negative_annotations[pid])


        for pid in aapm_patient_ids:

            if pid in id2positive_annotations_aapm:

                self.id2positive_annotations[pid] = id2positive_annotations_aapm[pid]

                self.id2negative_annotations[pid] = id2negative_annotations_aapm[pid]
                self.patient_paths.append((pid,utils_lung.get_path_to_image_from_patient(aapm_data_path,pid)))


                n_positive += len(id2positive_annotations[pid])
                n_negative += len(id2negative_annotations[pid])

        print 'n positive', n_positive
        print 'n negative', n_negative

        self.nsamples = len(self.patient_paths)

        print 'n patients', self.nsamples

        self.aapm_patient_ids=aapm_patient_ids

        self.data_path = data_path
        self.batch_size = batch_size
        self.rng = rng
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.positive_proportion = positive_proportion


    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')

                y_batch = np.zeros((nb, 3), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    (id,patient_path) = self.patient_paths[idx]

                    
                    patients_ids.append(id)


                    
 
                    
                    if '.mhd' in patient_path:
                        img, origin, pixel_spacing = utils_lung.read_mhd(patient_path)
                        world_coord_system=True
                    else:
                        img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)
                        origin=None
                        world_coord_system=False


                    if i < np.rint(self.batch_size * self.positive_proportion):
                        patient_annotations = self.id2positive_annotations[id]
                        y_batch[i,0]=1
                    else:
                        patient_annotations = self.id2negative_annotations[id]
                        y_batch[i,0]=0


                    random_index=self.rng.randint(len(patient_annotations))
                    patch_center = patient_annotations[random_index]

                    if y_batch[i,0]==1 and id in self.aapm_patient_ids:
                        y_batch[i,1]=1
                        y_batch[i,2]=patch_center[-1]
                    
              
                    x_batch[i, 0, :, :, :] = self.data_prep_fun(data=img,
                                                                patch_center=patch_center,
                                                                pixel_spacing=pixel_spacing,
                                                                luna_origin=origin,
                                                                world_coord_system=world_coord_system)


                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:

                break





class DSBPatientsDataGeneratorTrainPlusTest(object):
    def __init__(self, data_path, batch_size, transform_params, id2candidates_path, data_prep_fun,
                 n_candidates_per_patient, rng, random, infinite, shuffle_top_n=False, patient_ids=None):

        id2label_list = utils_lung.read_labels(pathfinder.LABELS_PATH).items() # train labels
        id2label_list.extend(utils_lung.read_test_labels(pathfinder.TEST_LABELS_PATH).items()) # test labels
        self.id2label = dict(id2label_list)

        print("Length labels: "+str(len(self.id2label)))

        self.id2candidates_path = id2candidates_path
        self.patient_paths = []
        if patient_ids is not None:
            for pid in patient_ids:
                if pid in self.id2candidates_path:  # TODO: this should be redundant if fpr and segemntation are correctly generated
                    self.patient_paths.append(data_path + '/' + pid)
        else:
            raise ValueError('provide patient ids')

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.batch_size = batch_size
        self.transform_params = transform_params
        self.n_candidates_per_patient = n_candidates_per_patient
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.shuffle_top_n = shuffle_top_n

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)

            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]

                x_batch = np.zeros((self.batch_size*self.n_candidates_per_patient,)
                                   + self.transform_params['patch_size'], dtype='float32')
                x_batch_tmp = np.zeros((self.batch_size, self.n_candidates_per_patient, 1,)
                                   + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((self.batch_size,), dtype='float32')
                pids_batch = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    pid = utils_lung.extract_pid_dir(patient_path)

                    img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

                    all_candidates = utils.load_pkl(self.id2candidates_path[pid])
                    top_candidates = all_candidates[:self.n_candidates_per_patient]
                    if self.shuffle_top_n:
                        self.rng.shuffle(top_candidates)

                    x_batch_tmp[i] = np.float32(self.data_prep_fun(data=img,
                                                               patch_centers=top_candidates,
                                                               pixel_spacing=pixel_spacing))[:, None, :, :, :]
                    y_batch[i] = self.id2label[pid]
                    pids_batch.append(pid)
                
                    
                if len(idxs_batch) == self.batch_size:
                    x_batch=x_batch_tmp.reshape((self.batch_size*self.n_candidates_per_patient,)+self.transform_params['patch_size'])
                    yield x_batch, y_batch, pids_batch

            if not self.infinite:
                break

class DSBPatientsDataGeneratorTestWithLabels(object):
    def __init__(self, data_path, batch_size, transform_params, id2candidates_path, data_prep_fun,
                 n_candidates_per_patient, rng, random, infinite, shuffle_top_n=False, patient_ids=None):

        self.id2label = utils_lung.read_test_labels(pathfinder.TEST_LABELS_PATH)
        self.id2candidates_path = id2candidates_path
        self.patient_paths = []
        if patient_ids is not None:
            for pid in patient_ids:
                if pid in self.id2candidates_path:  # TODO: this should be redundant if fpr and segemntation are correctly generated
                    self.patient_paths.append(data_path + '/' + pid)
        else:
            raise ValueError('provide patient ids')

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.batch_size = batch_size
        self.transform_params = transform_params
        self.n_candidates_per_patient = n_candidates_per_patient
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.shuffle_top_n = shuffle_top_n

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)

            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]

                x_batch = np.zeros((self.batch_size, self.n_candidates_per_patient, 1,)
                                   + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((self.batch_size,), dtype='float32')
                pids_batch = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    pid = utils_lung.extract_pid_dir(patient_path)

                    img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

                    all_candidates = utils.load_pkl(self.id2candidates_path[pid])
                    top_candidates = all_candidates[:self.n_candidates_per_patient]
                    if self.shuffle_top_n:
                        self.rng.shuffle(top_candidates)

                    x_batch[i] = np.float32(self.data_prep_fun(data=img,
                                                               patch_centers=top_candidates,
                                                               pixel_spacing=pixel_spacing))[:, None, :, :, :]
                    y_batch[i] = self.id2label[pid]
                    pids_batch.append(pid)

                if len(idxs_batch) == self.batch_size:
                    yield x_batch, y_batch, pids_batch

            if not self.infinite:
                break

class DSBPatientsDataHeatmapGenerator(object):
    def __init__(self, data_path, transform_params,batch_size, id2heatmap_path, data_prep_fun,
                  rng, random, infinite, patient_ids=None):

        self.id2label = utils_lung.read_labels(pathfinder.LABELS_PATH)
        self.id2heatmap_path = id2heatmap_path
        self.patient_paths = []
        if patient_ids is not None:
            for pid in patient_ids:
                if pid in self.id2heatmap_path:  # TODO: this should be redundant if fpr and segemntation are correctly generated
                    self.patient_paths.append(data_path + '/' + pid)
        else:
            raise ValueError('provide patient ids')

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.batch_size = batch_size
        self.transform_params = transform_params
        self.rng = rng
        self.random = random
        self.infinite = infinite


    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)

            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]

                x_batch = np.zeros((self.batch_size,) + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((self.batch_size,), dtype='float32')
                pids_batch = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    pid = utils_lung.extract_pid_dir(patient_path)

                    heatmap = utils.load_pkl(self.id2heatmap_path[pid])

                    x_batch[i] = self.data_prep_fun(data=heatmap)
                    y_batch[i] = self.id2label[pid]
                    pids_batch.append(pid)

                if len(idxs_batch) == self.batch_size:
                    yield x_batch, y_batch, pids_batch

            if not self.infinite:
                break


class DSBPatientsDataGeneratorTest(object):
    def __init__(self, data_path, batch_size, transform_params, id2candidates_path, data_prep_fun,
                 n_candidates_per_patient, rng, random, infinite, shuffle_top_n=False, patient_ids=None):

        self.id2label = utils_lung.read_labels(pathfinder.LABELS_PATH)
        self.id2candidates_path = id2candidates_path
        self.patient_paths = []
        if patient_ids is not None:
            for pid in patient_ids:
                if pid in self.id2candidates_path:  # TODO: this should be redundant if fpr and segemntation are correctly generated
                    self.patient_paths.append(data_path + '/' + pid)
                else:
                    print("not in here!")
        else:
            raise ValueError('provide patient ids')

        self.nsamples = len(self.patient_paths)
        self.data_path = data_path
        self.data_prep_fun = data_prep_fun
        self.batch_size = batch_size
        self.transform_params = transform_params
        self.n_candidates_per_patient = n_candidates_per_patient
        self.rng = rng
        self.random = random
        self.infinite = infinite
        self.shuffle_top_n = shuffle_top_n

    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)

            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]

                x_batch_tmp = np.zeros((self.batch_size, self.n_candidates_per_patient,)
                                   + self.transform_params['patch_size'], dtype='float32')

                x_batch = np.zeros((self.batch_size*self.n_candidates_per_patient,)
                                   + self.transform_params['patch_size'], dtype='float32')
                y_batch = np.zeros((self.batch_size,), dtype='float32')
                pids_batch = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]
                    pid = utils_lung.extract_pid_dir(patient_path)

                    img, pixel_spacing = utils_lung.read_dicom_scan(patient_path)

                    all_candidates = utils.load_pkl(self.id2candidates_path[pid])
                    top_candidates = all_candidates[:self.n_candidates_per_patient]
                    if self.shuffle_top_n:
                        self.rng.shuffle(top_candidates)

                    x_batch_tmp[i] = np.float32(self.data_prep_fun(data=img,
                                                               patch_centers=top_candidates,
                                                               pixel_spacing=pixel_spacing))[:, :, :, :]
                    #y_batch[i] = self.id2label[pid]
                    pids_batch.append(pid)

                if len(idxs_batch) == self.batch_size:
                    x_batch = x_batch_tmp.reshape((self.batch_size * self.n_candidates_per_patient,)
                                   + self.transform_params['patch_size'])
                    yield x_batch, None, pids_batch

            if not self.infinite:
                break



class CandidatesLunaSizeDataGenerator(object):
    def __init__(self, data_path, batch_size, transform_params, patient_ids, data_prep_fun, rng,
                 full_batch, random, infinite, positive_proportion, **kwargs):

        id2positive_annotations = utils_lung.read_luna_annotations(pathfinder.LUNA_LABELS_PATH)
        id2negative_annotations = utils_lung.read_luna_negative_candidates(pathfinder.LUNA_CANDIDATES_PATH)

        self.file_extension = '.pkl' if 'pkl' in data_path else '.mhd'
        self.id2positive_annotations = {}
        self.id2negative_annotations = {}
        self.patient_paths = []
        n_positive, n_negative = 0, 0
        for pid in patient_ids:
            if pid in id2positive_annotations:
                self.id2positive_annotations[pid] = id2positive_annotations[pid]
                self.id2negative_annotations[pid] = id2negative_annotations[pid]
                self.patient_paths.append(data_path + '/' + pid + self.file_extension)

                n_positive += len(id2positive_annotations[pid])
                n_negative += len(id2negative_annotations[pid])

        print 'n positive', n_positive
        print 'n negative', n_negative

        self.nsamples = len(self.patient_paths)

        print 'n patients', self.nsamples

        self.data_path = data_path
        self.batch_size = batch_size
        self.rng = rng
        self.full_batch = full_batch
        self.random = random
        self.infinite = infinite
        self.data_prep_fun = data_prep_fun
        self.transform_params = transform_params
        self.positive_proportion = positive_proportion


    def generate(self):
        while True:
            rand_idxs = np.arange(self.nsamples)
            if self.random:
                self.rng.shuffle(rand_idxs)
            for pos in xrange(0, len(rand_idxs), self.batch_size):
                idxs_batch = rand_idxs[pos:pos + self.batch_size]
                nb = len(idxs_batch)
                # allocate batches
                x_batch = np.zeros((nb, 1) + self.transform_params['patch_size'], dtype='float32')

                y_batch = np.zeros((nb, 1), dtype='float32')
                patients_ids = []

                for i, idx in enumerate(idxs_batch):
                    patient_path = self.patient_paths[idx]

                    id = utils_lung.extract_pid_filename(patient_path, self.file_extension)
                    patients_ids.append(id)

                    img, origin, pixel_spacing = utils_lung.read_pkl(patient_path) \
                        if self.file_extension == '.pkl' else utils_lung.read_mhd(patient_path)
                    if i < np.rint(self.batch_size * self.positive_proportion):
                        patient_annotations = self.id2positive_annotations[id]
                    else:
                        patient_annotations = self.id2negative_annotations[id]

                    patch_center = patient_annotations[self.rng.randint(len(patient_annotations))]

                    y_batch[i] = float(patch_center[-1])
                    x_batch[i, 0, :, :, :] = self.data_prep_fun(data=img,
                                                                patch_center=patch_center,
                                                                pixel_spacing=pixel_spacing,
                                                                luna_origin=origin)

                if self.full_batch:
                    if nb == self.batch_size:
                        yield x_batch, y_batch, patients_ids
                else:
                    yield x_batch, y_batch, patients_ids

            if not self.infinite:

                break
