from os import listdir
from os.path import isfile, join
import numpy as np
from math import floor
from scipy.ndimage.interpolation import zoom, rotate

import imageio
import face_recognition

from sklearn.model_selection import train_test_split
import os
import cv2

from keras.models import Model as KerasModel
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, LeakyReLU
from keras.optimizers import Adam




## Face extraction
class Video:
    def __init__(self, path):
        self.path = path
        self.container = imageio.get_reader(path, 'ffmpeg')
        self.length = self.container.count_frames()
        self.fps = self.container.get_meta_data()['fps']

    def init_head(self):
        self.container.set_image_index(0)

    def next_frame(self):
        self.container.get_next_data()

    def get(self, key):
        return self.container.get_data(key)

    def __call__(self, key):
        return self.get(key)

    def __len__(self):
        return self.length


class FaceFinder(Video):
    def __init__(self, path, load_first_face=True):
        super().__init__(path)
        self.faces = {}
        self.coordinates = {}  # stores the face (locations center, rotation, length)
        self.last_frame = self.get(0)
        self.frame_shape = self.last_frame.shape[:2]
        self.last_location = (0, 200, 200, 0)
        if (load_first_face):
            face_positions = face_recognition.face_locations(self.last_frame, number_of_times_to_upsample=2)
            if len(face_positions) > 0:
                self.last_location = face_positions[0]

    def load_coordinates(self, filename):
        np_coords = np.load(filename)
        self.coordinates = np_coords.item()

    def expand_location_zone(self, loc, margin=0.2):
        ''' Adds a margin around a frame slice '''
        offset = round(margin * (loc[2] - loc[0]))
        y0 = max(loc[0] - offset, 0)
        x1 = min(loc[1] + offset, self.frame_shape[1])
        y1 = min(loc[2] + offset, self.frame_shape[0])
        x0 = max(loc[3] - offset, 0)
        return (y0, x1, y1, x0)

    @staticmethod
    def upsample_location(reduced_location, upsampled_origin, factor):
        ''' Adapt a location to an upsampled image slice '''
        y0, x1, y1, x0 = reduced_location
        Y0 = round(upsampled_origin[0] + y0 * factor)
        X1 = round(upsampled_origin[1] + x1 * factor)
        Y1 = round(upsampled_origin[0] + y1 * factor)
        X0 = round(upsampled_origin[1] + x0 * factor)
        return (Y0, X1, Y1, X0)

    @staticmethod
    def pop_largest_location(location_list):
        max_location = location_list[0]
        max_size = 0
        if len(location_list) > 1:
            for location in location_list:
                size = location[2] - location[0]
                if size > max_size:
                    max_size = size
                    max_location = location
        return max_location

    @staticmethod
    def L2(A, B):
        return np.sqrt(np.sum(np.square(A - B)))

    def find_coordinates(self, landmark, K=2.2):
        '''
        We either choose K * distance(eyes, mouth),
        or, if the head is tilted, K * distance(eye 1, eye 2)
        /!\ landmarks coordinates are in (x,y) not (y,x)
        '''
        E1 = np.mean(landmark['left_eye'], axis=0)
        E2 = np.mean(landmark['right_eye'], axis=0)
        E = (E1 + E2) / 2
        N = np.mean(landmark['nose_tip'], axis=0) / 2 + np.mean(landmark['nose_bridge'], axis=0) / 2
        B1 = np.mean(landmark['top_lip'], axis=0)
        B2 = np.mean(landmark['bottom_lip'], axis=0)
        B = (B1 + B2) / 2

        C = N
        l1 = self.L2(E1, E2)
        l2 = self.L2(B, E)
        l = max(l1, l2) * K
        if (B[1] == E[1]):
            if (B[0] > E[0]):
                rot = 90
            else:
                rot = -90
        else:
            rot = np.arctan((B[0] - E[0]) / (B[1] - E[1])) / np.pi * 180

        return ((floor(C[1]), floor(C[0])), floor(l), rot)

    def find_faces(self, resize=0.5, stop=0, skipstep=0, no_face_acceleration_threshold=3, cut_left=0, cut_right=-1,
                   use_frameset=False, frameset=[]):
        '''
        The core function to extract faces from frames
        using previous frame location and downsampling to accelerate the loop.
        '''
        not_found = 0
        no_face = 0
        no_face_acc = 0

        # to only deal with a subset of a video, for instance I-frames only
        if (use_frameset):
            finder_frameset = frameset
        else:
            if (stop != 0):
                finder_frameset = range(0, min(self.length, stop), skipstep + 1)
            else:
                finder_frameset = range(0, self.length, skipstep + 1)

        # Quick face finder loop
        for i in finder_frameset:
            # Get frame
            frame = self.get(i)
            if (cut_left != 0 or cut_right != -1):
                frame[:, :cut_left] = 0
                frame[:, cut_right:] = 0

                # Find face in the previously found zone
            potential_location = self.expand_location_zone(self.last_location)
            potential_face_patch = frame[potential_location[0]:potential_location[2],
                                   potential_location[3]:potential_location[1]]
            potential_face_patch_origin = (potential_location[0], potential_location[3])

            reduced_potential_face_patch = zoom(potential_face_patch, (resize, resize, 1))
            reduced_face_locations = face_recognition.face_locations(reduced_potential_face_patch, model='cnn')

            if len(reduced_face_locations) > 0:
                no_face_acc = 0  # reset the no_face_acceleration mode accumulator

                reduced_face_location = self.pop_largest_location(reduced_face_locations)
                face_location = self.upsample_location(reduced_face_location,
                                                       potential_face_patch_origin,
                                                       1 / resize)
                self.faces[i] = face_location
                self.last_location = face_location

                # extract face rotation, length and center from landmarks
                landmarks = face_recognition.face_landmarks(frame, [face_location])
                if len(landmarks) > 0:
                    # we assume that there is one and only one landmark group
                    self.coordinates[i] = self.find_coordinates(landmarks[0])
            else:
                not_found += 1

                if no_face_acc < no_face_acceleration_threshold:
                    # Look for face in full frame
                    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
                else:
                    # Avoid spending to much time on a long scene without faces
                    reduced_frame = zoom(frame, (resize, resize, 1))
                    face_locations = face_recognition.face_locations(reduced_frame)

                if len(face_locations) > 0:
                    print('Face extraction warning : ', i, '- found face in full frame', face_locations)
                    no_face_acc = 0  # reset the no_face_acceleration mode accumulator

                    face_location = self.pop_largest_location(face_locations)

                    # if was found on a reduced frame, upsample location
                    if no_face_acc > no_face_acceleration_threshold:
                        face_location = self.upsample_location(face_location, (0, 0), 1 / resize)

                    self.faces[i] = face_location
                    self.last_location = face_location

                    # extract face rotation, length and center from landmarks
                    landmarks = face_recognition.face_landmarks(frame, [face_location])
                    if len(landmarks) > 0:
                        self.coordinates[i] = self.find_coordinates(landmarks[0])
                else:
                    print('Face extraction warning : ', i, '- no face')
                    no_face_acc += 1
                    no_face += 1

        # print('Face extraction report of', 'not_found :', not_found)
        # print('Face extraction report of', 'no_face :', no_face)
        return 0

    def get_face(self, i):
        ''' Basic unused face extraction without alignment '''
        frame = self.get(i)
        if i in self.faces:
            loc = self.faces[i]
            patch = frame[loc[0]:loc[2], loc[3]:loc[1]]
            return patch
        return frame

    @staticmethod
    def get_image_slice(img, y0, y1, x0, x1):
        '''Get values outside the domain of an image'''
        m, n = img.shape[:2]
        padding = max(-y0, y1 - m, -x0, x1 - n, 0)
        padded_img = np.pad(img, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
        return padded_img[(padding + y0):(padding + y1),
               (padding + x0):(padding + x1)]

    def get_aligned_face(self, i, l_factor=1.3):
        '''
        The second core function that converts the data from self.coordinates into an face image.
        '''
        frame = self.get(i)
        if i in self.coordinates:
            c, l, r = self.coordinates[i]
            l = int(l) * l_factor  # fine-tuning the face zoom we really want
            dl_ = floor(np.sqrt(2) * l / 2)  # largest zone even when rotated
            patch = self.get_image_slice(frame,
                                         floor(c[0] - dl_),
                                         floor(c[0] + dl_),
                                         floor(c[1] - dl_),
                                         floor(c[1] + dl_))
            rotated_patch = rotate(patch, -r, reshape=False)
            # note : dl_ is the center of the patch of length 2dl_
            return self.get_image_slice(rotated_patch,
                                        floor(dl_ - l // 2),
                                        floor(dl_ + l // 2),
                                        floor(dl_ - l // 2),
                                        floor(dl_ + l // 2))
        return frame


## Face prediction
class FaceBatchGenerator:
    '''
    Made to deal with framesubsets of video.
    '''

    def __init__(self, face_finder, target_size=256):
        self.finder = face_finder
        self.target_size = target_size
        self.head = 0
        self.length = int(face_finder.length)

    def resize_patch(self, patch):
        m, n = patch.shape[:2]
        return zoom(patch, (self.target_size / m, self.target_size / n, 1))

    def next_batch(self, batch_size=16):
        batch = np.zeros((1, self.target_size, self.target_size, 3))
        stop = min(self.head + batch_size, self.length)
        i = 0
        while (i < batch_size) and (self.head < self.length):
            if self.head in self.finder.coordinates:
                patch = self.finder.get_aligned_face(self.head)
                batch = np.concatenate((batch, np.expand_dims(self.resize_patch(patch), axis=0)),
                                       axis=0)
                i += 1
            self.head += 1
        return batch[1:] / 255.0


def predict_faces(generator, classifier, batch_size=50, output_size=1):
    '''
    Compute predictions for a face batch generator
    '''
    n = len(generator.finder.coordinates.items())
    profile = np.zeros((1, output_size))
    for epoch in range(n // batch_size + 1):
        face_batch = generator.next_batch(batch_size=batch_size)
        prediction = classifier.predict(face_batch)
        if (len(prediction) > 0):
            profile = np.concatenate((profile, prediction))
    return profile[1:]



# generate_images_from_video
def generate_images_from_video(dirname, save_root, frame_subsample_count=30, target_size=256):
    def resize_patch(patch):
        m, n = patch.shape[:2]
        return zoom(patch, (target_size / m, target_size / n, 1))

    '''
       Extraction + Prediction over a video
       '''
    filenames = [f for f in listdir(dirname) if
                 isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))]

    faces = np.zeros((1, target_size, target_size, 3))
    t = 0
    for vid in filenames:
        t += 1
        print('Dealing with ' + str(t) + 'th video :' + str(vid))

        # Compute face locations and store them in the face finder
        face_finder = FaceFinder(join(dirname, vid), load_first_face=False)
        skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
        face_finder.find_faces(resize=0.5, skipstep=skipstep)

        n = len(face_finder.coordinates.items())  # 检测到人脸个数
        length = int(face_finder.length)  # 视频中图片的长度
        i = 0
        head = 0
        while (i < n) and (head < length):
            if head in face_finder.coordinates:
                patch = face_finder.get_aligned_face(head)
                save_path = join(save_root, vid.split('.')[0] + '_' + str("%04d" % i) + '.jpg')
                print(save_path)
                imageio.imwrite(save_path, resize_patch(patch))
                faces = np.concatenate((faces, np.expand_dims(resize_patch(patch), axis=0)), axis=0)
                i += 1
            head += 1
    return faces[1:]


def compute_accuracy(classifier, dirname, frame_subsample_count=30):
    '''
    Extraction + Prediction over a video
    '''
    filenames = [f for f in listdir(dirname) if
                 isfile(join(dirname, f)) and ((f[-4:] == '.mp4') or (f[-4:] == '.avi') or (f[-4:] == '.mov'))]
    predictions = {}

    for vid in filenames:
        print('Dealing with video ', vid)

        # Compute face locations and store them in the face finder
        face_finder = FaceFinder(join(dirname, vid), load_first_face=False)
        skipstep = max(floor(face_finder.length / frame_subsample_count), 0)
        face_finder.find_faces(resize=0.5, skipstep=skipstep)

        print('Predicting ', vid)
        gen = FaceBatchGenerator(face_finder)
        p = predict_faces(gen, classifier)

        predictions[vid[:-4]] = (np.mean(p > 0.5), p)
    return predictions





IMGWIDTH = 256
class Classifier:

    def predict(self, x):
        return self.model.predict(x)

    def fit(self, x, y):
        return self.model.train_on_batch(x, y)

    def get_accuracy(self, x, y):
        return self.model.test_on_batch(x, y)

    def load(self, path):
        self.model.load_weights(path)


class Meso4(Classifier):
    def __init__(self, learning_rate=0.001):
        self.model = self.init_model()
        optimizer = Adam(lr=learning_rate)
        self.model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

    def init_model(self):
        x = Input(shape=(IMGWIDTH, IMGWIDTH, 3))

        x1 = Conv2D(8, (3, 3), padding='same', activation='relu')(x)
        x1 = BatchNormalization()(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

        x2 = Conv2D(8, (5, 5), padding='same', activation='relu')(x1)
        x2 = BatchNormalization()(x2)
        x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

        x3 = Conv2D(16, (5, 5), padding='same', activation='relu')(x2)
        x3 = BatchNormalization()(x3)
        x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

        x4 = Conv2D(16, (5, 5), padding='same', activation='relu')(x3)
        x4 = BatchNormalization()(x4)
        x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

        y = Flatten()(x4)
        y = Dropout(0.5)(y)
        y = Dense(16)(y)
        y = LeakyReLU(alpha=0.1)(y)
        y = Dropout(0.5)(y)
        y = Dense(1, activation='sigmoid')(y)

        return KerasModel(inputs=x, outputs=y)

def get_face(self):
    dirname_real = 'C:/Users/Donny/Desktop/video/real'
    save_root_real = 'C:/Users/Donny/Desktop/face_image/real'

    dirname_synthesis = 'C:/Users/Donny/Desktop/video/synthesis'
    save_root_synthesis = 'C:/Users/Donny/Desktop/face_image/synthesis'

    generate_images_from_video(dirname_real, save_root_real, frame_subsample_count=30, target_size=256)
    generate_images_from_video(dirname_synthesis, save_root_synthesis, frame_subsample_count=30, target_size=256)

    x_train = []
    y_train = []

    path_0 = 'C:/Users/Donny/Desktop/CelebDF/face_image/synthesis'
    path_1 = 'C:/Users/Donny/Desktop/CelebDF/face_image/real'
    filename_0 = os.listdir(path_0)
    filename_1 = os.listdir(path_1)

    for i in range(len(filename_0)):
        image_path = 'C:/Users/Donny/Desktop/CelebDF/face_image/synthesis/' + filename_0[i]

        x = cv2.imread(image_path)
        x = np.array(x)
        x = cv2.resize(x, (256, 256))
        x = x.reshape(256, 256, 3)
        x_train.append(x)
        y_train.append(0)

    for i in range(len(filename_1)):
        image_path = 'C:/Users/Donny/Desktop/CelebDF/face_image/real/' + filename_1[i]

        x = cv2.imread(image_path)
        x = np.array(x)
        x = cv2.resize(x, (256, 256))
        x = x.reshape(256, 256, 3)
        x_train.append(x)
        y_train.append(1)

    x_train = np.array(x_train)
    y_train = np.array(y_train)


def train(self,x_train, y_train):
    # 训练
    classifier = Meso4()
    history = classifier.model.fit(x_train,
                                   y_train,
                                   batch_size=32,
                                   epochs=10)


class Model():

    def run(input_dir):
        # 预测视频
        classifier = Meso4()
        classifier.model.load_weights('./weights/weight_1')
        predictions = compute_accuracy(classifier, input_dir)
        print(predictions)

        imgNames = []
        pre_label = []

        for video_name in predictions:
            imgNames.append(video_name)

            if predictions[video_name][0] > 0.5:
                label = 1
            if predictions[video_name][0] < 0.5:
                label = 0
            pre_label.append(label)

        return imgNames,pre_label


imgNames, pre_label = Model.run('C:/Users/Donny/Desktop/CelebDF/test_video')
print(imgNames)
print(pre_label)