"""
Generate feature vectors from the landmarks JSON file
and export them in a Numpy Zip file
"""
import json
import numpy as np


def label_to_vect(label_str):
    if label_str == 'SMILING':
        return 1
    return 0


def main():
    landmark_filename = 'data/landmarks.json'
    output_filename = 'data/smiling_data.npz'

    with open(landmark_filename, 'r') as f:
        landmark_data = json.load(f)

    sample_tags = list(landmark_data.keys())

    data_vectors = []
    data_labels = []

    for tag in sample_tags:
        _, sample_label = tag.split('_')

        if (sample_label not in {'SMILING', 'NEUTRAL'}):
            continue

        data_labels.append(label_to_vect(sample_label))

        data = landmark_data[tag]
        lip_upper_inner = np.array(data['lipsUpperInner'])
        lip_upper_outer = np.array(data['lipsUpperOuter'])
        lip_lower_inner = np.array(data['lipsLowerInner'])
        lip_lower_outer = np.array(data['lipsLowerOuter'])
        silhouette = np.array(data['silhouette'])

        avg_height_inner = np.abs(
            np.mean(lip_upper_inner, axis=0)[1]
            - np.mean(lip_lower_inner, axis=0)[1])

        avg_height_outer = np.abs(
            np.mean(lip_upper_outer, axis=0)[1]
            - np.mean(lip_lower_outer, axis=0)[1])

        width_upper_inner = np.abs(
            np.max(lip_upper_inner[:, 0])
            - np.min(lip_upper_inner[:, 0]))

        width_lower_inner = np.abs(
            np.max(lip_lower_inner[:, 0])
            - np.min(lip_lower_inner[:, 0]))

        width_upper_outer = np.abs(
            np.max(lip_upper_outer[:, 0])
            - np.min(lip_upper_outer[:, 0]))

        width_lower_outer = np.abs(
            np.max(lip_lower_outer[:, 0])
            - np.min(lip_lower_outer[:, 0]))

        avg_width_inner = (width_lower_inner + width_upper_inner) / 2.0
        avg_width_outer = (width_lower_outer + width_upper_outer) / 2.0

        face_width = np.abs(
            np.max(silhouette[:, 0]) - np.min(silhouette[:, 0]))
        face_height = np.abs(
            np.max(silhouette[:, 1]) - np.min(silhouette[:, 1]))

        data_vectors.append([
            avg_height_inner,
            avg_height_outer,
            avg_width_inner,
            avg_width_outer,
            face_width,
            face_height
        ])

    np.savez(output_filename, data=np.array(
        data_vectors), labels=np.array(data_labels, dtype=int))


if __name__ == '__main__':
    main()
