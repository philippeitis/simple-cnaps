import torch
import numpy as np
import argparse
import os
import sys
import random
from utils import print_and_log, get_log_files, ValidationAccuracies, loss, aggregate_accuracy
from model import SimpleCnaps
from meta_dataset_reader import MetaDatasetReader
from collections import Counter
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Quiet TensorFlow warnings
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # Quiet TensorFlow warnings

NUM_TRAIN_TASKS = 110000
NUM_VALIDATION_TASKS = 200
NUM_TEST_TASKS = 600
VALIDATION_FREQUENCY = 10000


def main():
    tf.compat.v1.disable_eager_execution()
    learner = Learner()
    learner.run()


class Learner:
    def __init__(self):
        self.args = self.parse_command_line()

        self.checkpoint_dir, self.logfile, self.checkpoint_path_validation, self.checkpoint_path_final \
            = get_log_files(self.args.checkpoint_dir)

        print_and_log(self.logfile, "Options: %s\n" % self.args)
        print_and_log(self.logfile, "Checkpoint Directory: %s\n" % self.checkpoint_dir)

        gpu_device = f'cuda:{self.args.gpu}'
        self.device = torch.device(gpu_device if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model()
        self.train_set, self.validation_set, self.test_set = self.init_data()
        self.metadataset = MetaDatasetReader(self.args.data_path, self.args.mode, self.train_set, self.validation_set,
                                             self.test_set)
        self.loss = loss
        self.accuracy_fn = aggregate_accuracy
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer.zero_grad()
        self.validation_accuracies = ValidationAccuracies(self.validation_set)

    def init_model(self):
        use_two_gpus = self.use_two_gpus()
        model = SimpleCnaps(device=self.device, use_two_gpus=use_two_gpus, args=self.args).to(self.device)
        model.train()  # set encoder is always in train mode to process context data
        model.feature_extractor.eval()  # feature extractor is always in eval mode
        if use_two_gpus:
            model.distribute_model()
        return model

    @staticmethod
    def init_data():
        # train_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower']
        # validation_set = ['ilsvrc_2012', 'omniglot', 'aircraft', 'cu_birds', 'dtd', 'quickdraw', 'fungi', 'vgg_flower',
        #                   'mscoco']
        test_set = [
            # 'ilsvrc_2012',
            'omniglot', 'aircraft',
            # 'cu_birds',
            'dtd',
            # 'quickdraw',
            'fungi', 'vgg_flower', 'traffic_sign',
                 #'mscoco',
       'mnist', 'cifar10', 'cifar100']

        train_set = ['dtd']
        validation_set = []
       # test_set = ['mnist', 'cifar10', 'cifar100']

        return train_set, validation_set, test_set

    @staticmethod
    def parse_command_line():
        """
        Command line parser
        """
        parser = argparse.ArgumentParser()

        parser.add_argument("--data_path", default="../datasets", help="Path to dataset records.")
        parser.add_argument("--pretrained_resnet_path", default="../models/pretrained_resnet.pt.tar",
                            help="Path to pretrained feature extractor model.")
        parser.add_argument("--mode", choices=["train", "test", "train_test"], default="train_test",
                            help="Whether to run training only, testing only, or both training and testing.")
        parser.add_argument("--learning_rate", "-lr", type=float, default=5e-4, help="Learning rate.")
        parser.add_argument("--tasks_per_batch", type=int, default=16,
                            help="Number of tasks between parameter optimizations.")
        parser.add_argument("--checkpoint_dir", "-c", default='../checkpoints', help="Directory to save checkpoint to.")
        parser.add_argument("--test_model_path", "-m", default=None, help="Path to model to load and test.")
        parser.add_argument("--feature_adaptation", choices=["no_adaptation", "film", "film+ar"], default="film+ar",
                            help="Method to adapt feature extractor parameters.")
        parser.add_argument("--gpu", default=0, type=int, help="GPU index.")
        parser.add_argument("--augment", default=5, type=int, help="Number of additional points to augment.")

        args = parser.parse_args()

        return args

    def run(self):
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.compat.v1.Session(config=config) as session:
            if self.args.mode == 'train' or self.args.mode == 'train_test':
                train_accuracies = []
                losses = []
                # Rounds up to next multiple so we log every n batches instead.
                total_iterations = (NUM_TRAIN_TASKS + self.args.tasks_per_batch + 1) // self.args.tasks_per_batch
                num_iters = NUM_TRAIN_TASKS // self.args.tasks_per_batch
                validation_iters = VALIDATION_FREQUENCY // self.args.tasks_per_batch
                log_iters = 1000 // self.args.tasks_per_batch

                for iteration in range(num_iters):
                    # Do batch iters.
                    for _ in range(self.args.tasks_per_batch):
                        torch.set_grad_enabled(True)
                        task_dict = self.metadataset.get_train_task(session)
                        task_loss, task_accuracy = self.train_task(task_dict)
                        train_accuracies.append(task_accuracy)
                        losses.append(task_loss)

                        # optimize
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    # Log every so often.
                    if (iteration + 1) % log_iters == 0:
                        # print training stats
                        print_and_log(self.logfile, 'Task [{}/{}], Train Loss: {:.7f}, Train Accuracy: {:.7f}'
                                      .format(iteration + 1, total_iterations, torch.Tensor(losses).mean().item(),
                                              torch.Tensor(train_accuracies).mean().item()))
                        train_accuracies = []
                        losses = []

                    # Validate when necessary.
                    if ((iteration + 1) % validation_iters == 0) and (iteration + 1) != num_iters:
                        # validate
                        accuracy_dict = self.validate(session)
                        self.validation_accuracies.print(self.logfile, accuracy_dict)
                        # save the model if validation is the best so far
                        if self.validation_accuracies.is_better(accuracy_dict):
                            self.validation_accuracies.replace(accuracy_dict)
                            torch.save(self.model.state_dict(), self.checkpoint_path_validation)
                            print_and_log(self.logfile, 'Best validation model was updated.\n')

                # save the final model
                torch.save(self.model.state_dict(), self.checkpoint_path_final)

            if self.args.mode == 'train_test':
                self.test(self.checkpoint_path_final, session)
                self.test(self.checkpoint_path_validation, session)

            if self.args.mode == 'test':
                self.test(self.args.test_model_path, session)

            self.logfile.close()

    def train_task(self, task_dict):
        context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)

        for i in range(0, 5):
            target_logits = self.model(context_images, context_labels, target_images)
            task_loss = self.loss(target_logits, target_labels, self.device) / self.args.tasks_per_batch
            if self.args.feature_adaptation == 'film' or self.args.feature_adaptation == 'film+ar':
                if self.use_two_gpus():
                    regularization_term = (self.model.feature_adaptation_network.regularization_term()).cuda(0)
                else:
                    regularization_term = (self.model.feature_adaptation_network.regularization_term())
                regularizer_scaling = 0.001
                task_loss += regularizer_scaling * regularization_term
            task_accuracy = self.accuracy_fn(target_logits, target_labels)

            task_loss.backward(retain_graph=False)
            if target_logits.entropy() < 0.10:
                break

        return task_loss, task_accuracy

    def validate(self, session):
        with torch.no_grad():
            accuracy_dict = {}
            for item in self.validation_set:
                accuracies = []
                for _ in range(NUM_VALIDATION_TASKS):
                    task_dict = self.metadataset.get_validation_task(item, session)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                    target_logits = self.model(context_images, context_labels, target_images)
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())
                    del target_logits

                accuracy = np.array(accuracies).mean() * 100.0
                confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                accuracy_dict[item] = {"accuracy": accuracy, "confidence": confidence}

        return accuracy_dict

    def test(self, path, _session):
        self.model = self.init_model()
        self.model.load_state_dict(torch.load(path))
        print_and_log(self.logfile, f"\nTesting model {path}: ")  # add a blank line

        with torch.no_grad():
            for item in self.test_set:
                accuracies = []
                for _ in range(NUM_TEST_TASKS):
                    task_dict = self.metadataset.get_test_task(item, _session)
                    context_images, target_images, context_labels, target_labels = self.prepare_task(task_dict)
                    selected_class_indices = {}
                    unselected_class_indices = {}
                    random_select = {}
                    random_unselect = {}
                    for c in torch.unique(context_labels):
                        class_mask = torch.eq(context_labels, c)  # binary mask of labels equal to which_class
                        class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
                        # reshape to be a 1D vector
                        indices = torch.torch.reshape(class_mask_indices, (-1,)).tolist()
                        length = max(len(indices) // 10, min(len(indices), 1))
                        selected_class_indices[c.item()] = indices[:length]
                        unselected_class_indices[c.item()] = indices[length:]
                        indices = list(indices)
                        random.shuffle(indices)
                        random_select[c.item()] = indices[:length]
                        random_unselect[c.item()] = indices[length:]
                        del c

                    # print(selected_class_indices)
                    i = 15
                    target_logits, selected_indices, selected_images, selected_labels = None, None, None, None
                    while True:
                        if target_logits is not None:
                            del selected_indices
                            del selected_images
                            del selected_labels
                            del target_logits

                        selected_indices = []
                        for indices in selected_class_indices.values():
                            selected_indices.extend(indices)

                        selected_indices = torch.LongTensor(selected_indices).to(self.device)
                        selected_images = torch.index_select(context_images, 0, selected_indices)
                        selected_labels = torch.index_select(context_labels, 0, selected_indices)
                        target_logits = self.model(selected_images, selected_labels, target_images)

                        i -= 1

                        entropy_sum = 0.
                        for row in torch.transpose(target_logits, 0, 1):
                       #      print(entropy_sum, row, target_logits.shape)
                            entropy_sum += torch.distributions.Categorical(logits=row).entropy().item()

                        if entropy_sum < 0.25 * len(selected_indices) or i == 0:
                            break

                        weights = {}
                        for classx in selected_class_indices.keys():
                            if unselected_class_indices[classx]:
                                class_mask = torch.eq(selected_labels, classx)  # binary mask of labels equal to which_class
                                class_mask_indices = torch.torch.reshape(torch.nonzero(class_mask), (-1,))  # indices of labels equal to which class
                                # reshape to be a 1D vector

                                class_var = torch.index_select(
                                    self.model.context_features, 0, class_mask_indices
                                ).var()
                                # Close to 1 for low # of items
                                size_regularizer = 1. / ((class_mask_indices.shape[0]/4.) ** 2 + 1.0)
                                weights[classx] = (torch.sum(target_logits[:, classx, :]) * (class_var + size_regularizer)).item() #, class_var)
                        if not weights:
                            break
                        # Exploration
#                        if random.random() > np.exp(-i/5.):
#                            classx, tup = min(weights.items(), key=lambda x: x[1])
#                        else:
#                            classx, tup = random.choice(list(weights.items()))
                        context_size = max(context_images.shape[0] // 20, 1)
                        sampling_from = list(weights.keys()) 
                        unread = sum([len(x) for x in unselected_class_indices.values()])
                        if unread > context_size:
                            to_sample = context_size
                            while to_sample != 0:
                                values = list(weights.values())
                                #print(values)
                                values = np.exp(np.array(values) / min(values))
                                values /= sum(values)
                                dist = torch.distributions.Categorical(probs=torch.FloatTensor(list(values)))
                                vals = dist.sample(torch.Size([to_sample]))
                                #print(vals, weights, to_sample, dist.probs)
                                for val in vals:
                                    indices = unselected_class_indices[sampling_from[val.item()]]
                                    if indices:
                                        to_sample -= 1
                                        selected_class_indices[classx].append(indices.pop())
                            select_left = context_size
                            while select_left != 0:
                                index = np.random.randint(len(selected_class_indices))
                                if random_unselect[index]:
                                    random_select[index].append(random_unselect[index].pop())
                                    select_left -= 1
                        else:
                            for classx, indices in unselected_class_indices.items():
                                selected_class_indices[classx].extend(indices)
                                indices.clear()
                                random_select[classx].extend(random_unselect[classx])
                                random_unselect[classx].clear()

                        # indices = unselected_class_indices[classx]
                               # length = max(int((first_len + len(indices)) * percentage), min(len(indices), 1))
                        #del unselected_class_indices[classx][:length]
                        # var = tup[1]
                        # Add up to 40% of available images depending on variance
                        #percentage = 0.4 / (1.0 + np.exp(-var.item()))

                        # print(l1, len(unselected_class_indices[classx]))
                    accuracy = self.accuracy_fn(target_logits, target_labels)
                    accuracies.append(accuracy.item())

                    fractions = [0, 0, 1/4, 1/2, 3/4, 1]
                    fraction_indices = [[] for _ in fractions]
                    for classx, indices in selected_class_indices.items():
                        all_indices = indices + unselected_class_indices[classx]
                        for i, fraction in enumerate(fractions):
                            random.shuffle(all_indices)
                            fraction_indices[i] += all_indices[:max(int(len(all_indices) * fraction), 1)]
                    fraction_indices[0] = [item for sublist in random_select.values() for item in sublist]
                    fraction_indices = [torch.LongTensor(fi).to(self.device) for fi in fraction_indices]
                    fraction_images = [torch.index_select(context_images, 0, fi) for fi in fraction_indices]
                    fraction_labels = [torch.index_select(context_labels, 0, fi) for fi in fraction_indices]
                    fraction_logits = [self.model(images, labels, target_images) for images, labels in zip(fraction_images, fraction_labels)]
                    fraction_accuracy = [self.accuracy_fn(logits, target_labels) for logits in fraction_logits]
                    fraction_names = ["RPAR", "SING", "QUAR", "HALF", "TQUA", "FULL"]

                    selections = [
                        ("PART", accuracy.item(), selected_indices.shape[0]),
                    ] + [(name, acc.item(), ims.shape[0]) for name, acc, ims in zip(fraction_names, fraction_accuracy, fraction_images)]

                    # print(context_images.shape)
                    print(f"{selections[0][0]}: {selections[0][1]:.4f} | {selections[0][2]:04} | {i:04} | {entropy_sum:07.3f}")
                    for selection in selections[1:]:
                        print(f"{selection[0]}: {selection[1]:.4f} | {selection[2]:04}")
                    selections = [(x[0], x[1] / x[2], x[1]) for x in selections]
                    selections.sort(key=lambda x: -x[1])
                    print(" ".join([f"{x[0]}: {x[1]:04f}" for x in selections]))
                    selections.sort(key=lambda x: -x[2])
                    print(" ".join([f"{x[0]}: {x[2]:04f}" for x in selections]))
                    for _ in range(len(fraction_indices)):
                        itemx = fraction_indices.pop()
                        del itemx
                        itemx = fraction_images.pop()
                        del itemx
                        itemx = fraction_labels.pop()
                        del itemx
                        itemx = fraction_logits.pop()
                        del itemx
                        itemx = fraction_accuracy.pop()
                        del itemx
                    del target_logits
                    del target_labels
                accuracy = np.array(accuracies).mean() * 100.0
                accuracy_confidence = (196.0 * np.array(accuracies).std()) / np.sqrt(len(accuracies))

                print_and_log(self.logfile, '{0:}: {1:3.1f}+/-{2:2.1f}'.format(item, accuracy, accuracy_confidence))

    def prepare_task(self, task_dict):
        context_images_np, context_labels_np = task_dict['context_images'], task_dict['context_labels']
        target_images_np, target_labels_np = task_dict['target_images'], task_dict['target_labels']

        context_images_np = context_images_np.transpose([0, 3, 1, 2])
        context_images_np, context_labels_np = self.shuffle(context_images_np, context_labels_np)
        context_images = torch.from_numpy(context_images_np)
        context_labels = torch.from_numpy(context_labels_np)

        target_images_np = target_images_np.transpose([0, 3, 1, 2])
        target_images_np, target_labels_np = self.shuffle(target_images_np, target_labels_np)
        target_images = torch.from_numpy(target_images_np)
        target_labels = torch.from_numpy(target_labels_np)

        context_images = context_images.to(self.device)
        target_images = target_images.to(self.device)
        context_labels = context_labels.to(self.device)
        target_labels = target_labels.type(torch.LongTensor).to(self.device)

        return context_images, target_images, context_labels, target_labels

    def shuffle(self, images, labels):
        """
        Return shuffled data.
        """
        permutation = np.random.permutation(images.shape[0])
        return images[permutation], labels[permutation]

    def use_two_gpus(self):
        # film+ar model does not fit on one GPU, so use model parallelism
        return self.args.feature_adaptation == "film+ar"


if __name__ == "__main__":
    main()
