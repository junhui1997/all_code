import argparse
import os
import cv2


def main(data_dir):
    LOSO_splits = ['1', '2', '3', '4', '5']
    LOUO_splits = ['B', 'C', 'D', 'E', 'F', 'G', 'H', 'I']
    out_dir = "./Splits"

    for task in ["Suturing", "Needle_Passing", "Knot_Tying"]:
        meta_file = os.path.join(data_dir, task, "meta_file_{}.txt".format(task))
        _annotations = [x.strip().split('\t') for x in open(meta_file)]
        annotations = []
        for i in range(len(_annotations)):
            annotation = []
            for elem in _annotations[i]:
                if elem:
                    annotation.append(elem)
            if annotation:
                annotations.append(annotation)
        trials = [row[0].split('_')[-1] for row in annotations]

        video_frame_counts = {}
        for trial in trials:
            video_file = os.path.join(data_dir, task, "video", "{}_{}_capture2.avi".format(task, trial))
            cap = cv2.VideoCapture(video_file)
            video_frame_counts[trial] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

        for splits in [LOSO_splits, LOUO_splits]:
            eval_scheme = "LOSO" if len(splits) == len(LOSO_splits) else "LOUO"
            for i in range(len(splits)):
                split = [trial for trial in trials if splits[i] in trial]
                if not os.path.exists(os.path.join(out_dir, task, eval_scheme)):
                    os.makedirs(os.path.join(out_dir, task, eval_scheme))
                split_file = open(os.path.join(out_dir, task, eval_scheme, "data_{}.csv".format(splits[i])), mode='w')
                for trial in sorted(split):
                    row = ["{}_{}".format(task, trial)]
                    row.append(video_frame_counts[trial])

                    row = [str(elem) for elem in row]
                    split_file.write(','.join(row) + os.linesep)

                split_file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create video list files.")
    parser.add_argument('data_dir', type=str, help="Path to data folder, which contains the extracted images "
                                                   "for each video. One subfolder per video.")

    args = parser.parse_args()
    main(args.data_dir)
