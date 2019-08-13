from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

import horovod.tensorflow as hvd

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')


def main(_argv):
    # Horovod: initialize Horovod.
    hvd.init()

    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    train_dataset = dataset.load_fake_dataset()
    if FLAGS.dataset:
        train_dataset = dataset.load_tfrecord_dataset(
            FLAGS.dataset, FLAGS.classes)
    train_dataset = train_dataset.shuffle(buffer_size=1024)  # TODO: not 1024
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = dataset.load_fake_dataset()
    if FLAGS.val_dataset:
        val_dataset = dataset.load_tfrecord_dataset(
            FLAGS.val_dataset, FLAGS.classes)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))

    if FLAGS.transfer != 'none':
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.mode == 'frozen':
            # freeze everything
            freeze_all(model)
        else:
            # reset top layers
            if FLAGS.tiny:  # get initial weights
                init_model = YoloV3Tiny(FLAGS.size, training=True)
            else:
                init_model = YoloV3(FLAGS.size, training=True)

            if FLAGS.transfer == 'darknet':
                for l in model.layers:
                    if l.name != 'yolo_darknet' and l.name.startswith('yolo_'):
                        l.set_weights(init_model.get_layer(
                            l.name).get_weights())
                    else:
                        freeze_all(l)
            elif FLAGS.transfer == 'no_output':
                for l in model.layers:
                    if l.name.startswith('yolo_output'):
                        l.set_weights(init_model.get_layer(
                            l.name).get_weights())
                    else:
                        freeze_all(l)

    
    
    # Horovod: adjust learning rate based on number of GPUs.
    optimizer = tf.optimizers.Adam(FLAGS.learning_rate * hvd.size())
    # Horovod: add Horovod DistributedOptimizer.
    
    ###############################################
    loss = [YoloLoss(anchors[mask]) for mask in anchor_masks]

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset.take(5717//hvd.size())):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss
                # Horovod: add Horovod Distributed GradientTape.
                tape = hvd.DistributedGradientTape(tape)
                
                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                # Horovod: broadcast initial variable states from rank 0 to all other processes.
                # This is necessary to ensure consistent initialization of all workers when
                # training is started with random weights or restored from a checkpoint.
                #
                # Note: broadcast should be done after the first gradient step to ensure optimizer
                # initialization.
                if batch==0:
                    hvd.broadcast_variables(model.variables, root_rank=0)
                    hvd.broadcast_variables(optimizer.variables(), root_rank=0)

                #############################
                if hvd.rank()==0:
                        logging.info("{}_train_{}, {}, {}".format(
                            epoch, batch, total_loss.numpy(),
                            list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                ###########################
                avg_loss.update_state(total_loss)
            
            
            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss
                if hvd.rank()==0:
                    logging.info("{}_val_{}, {}, {}".format(
                        epoch, batch, total_loss.numpy(),
                        list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)
            if hvd.rank()==0:
                logging.info("{}, train: {}, val: {}".format(
                    epoch,
                    avg_loss.result().numpy(),
                    avg_val_loss.result().numpy()))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            if hvd.rank()==0:
                model.save_weights(
                    'checkpoints/horovod_yolov3_train_{}.tf'.format(epoch))
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
# from absl import app, flags, logging
# from absl.flags import FLAGS
# import tensorflow as tf
# import numpy as np
# import cv2
# from tensorflow.keras.callbacks import (
#     ReduceLROnPlateau,
#     EarlyStopping,
#     ModelCheckpoint,
#     TensorBoard
# )
# from yolov3_tf2.models import (
#     YoloV3, YoloV3Tiny, YoloLoss,
#     yolo_anchors, yolo_anchor_masks,
#     yolo_tiny_anchors, yolo_tiny_anchor_masks
# )
# from yolov3_tf2.utils import freeze_all
# import yolov3_tf2.dataset as dataset

# #for horovod training
# import horovod.tensorflow.keras as hvd


# flags.DEFINE_string('dataset', '', 'path to dataset')
# flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
# flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
# flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
#                     'path to weights file')
# flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
# flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
#                   'fit: model.fit, '
#                   'eager_fit: model.fit(run_eagerly=True), '
#                   'eager_tf: custom GradientTape')
# flags.DEFINE_enum('transfer', 'none',
#                   ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
#                   'none: Training from scratch, '
#                   'darknet: Transfer darknet, '
#                   'no_output: Transfer all but output, '
#                   'frozen: Transfer and freeze all, '
#                   'fine_tune: Transfer all and freeze darknet only')
# flags.DEFINE_integer('size', 416, 'image size')
# flags.DEFINE_integer('epochs', 2, 'number of epochs')
# flags.DEFINE_integer('batch_size', 8, 'batch size')
# flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')

# # def get_number(path):
# #     tf_records_filenames = path
# #     c = 0
# #     for record in tf.python_io.tf_record_iterator(fn):
# #         c += 1 
# #     return c


# def main(_argv):
#     # Horovod: initialize Horovod.
#     hvd.init()
#     # Horovod: pin GPU to be used to process local rank (one GPU per process)
#     gpus = tf.config.experimental.list_physical_devices('GPU')
#     for gpu in gpus:
#       tf.config.experimental.set_memory_growth(gpu, True)
#     if gpus:
#       tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')
#      #####################
#     if FLAGS.tiny:
#         model = YoloV3Tiny(FLAGS.size, training=True)
#         anchors = yolo_tiny_anchors
#         anchor_masks = yolo_tiny_anchor_masks
#     else:
#         model = YoloV3(FLAGS.size, training=True)
#         anchors = yolo_anchors
#         anchor_masks = yolo_anchor_masks

#     train_dataset = dataset.load_fake_dataset()
#     if FLAGS.dataset:
#         train_dataset = dataset.load_tfrecord_dataset(
#             FLAGS.dataset, FLAGS.classes)
#     train_dataset = train_dataset.shuffle(buffer_size=1024)  # TODO: not 1024
#     train_dataset = train_dataset.batch(FLAGS.batch_size)
#     train_dataset = train_dataset.map(lambda x, y: (
#         dataset.transform_images(x, FLAGS.size),
#         dataset.transform_targets(y, anchors, anchor_masks, 80)))
#     train_dataset = train_dataset.prefetch(
#         buffer_size=tf.data.experimental.AUTOTUNE)

#     val_dataset = dataset.load_fake_dataset()
#     if FLAGS.val_dataset:
#         val_dataset = dataset.load_tfrecord_dataset(
#             FLAGS.val_dataset, FLAGS.classes)
#     val_dataset = val_dataset.batch(FLAGS.batch_size)
#     val_dataset = val_dataset.map(lambda x, y: (
#         dataset.transform_images(x, FLAGS.size),
#         dataset.transform_targets(y, anchors, anchor_masks, 80)))

#     if FLAGS.transfer != 'none':
#         model.load_weights(FLAGS.weights)
#         if FLAGS.transfer == 'fine_tune':
#             # freeze darknet
#             darknet = model.get_layer('yolo_darknet')
#             freeze_all(darknet)
#         elif FLAGS.mode == 'frozen':
#             # freeze everything
#             freeze_all(model)
#         else:
#             # reset top layers
#             if FLAGS.tiny:  # get initial weights
#                 init_model = YoloV3Tiny(FLAGS.size, training=True)
#             else:
#                 init_model = YoloV3(FLAGS.size, training=True)

#             if FLAGS.transfer == 'darknet':
#                 for l in model.layers:
#                     if l.name != 'yolo_darknet' and l.name.startswith('yolo_'):
#                         l.set_weights(init_model.get_layer(
#                             l.name).get_weights())
#                     else:
#                         freeze_all(l)
#             elif FLAGS.transfer == 'no_output':
#                 for l in model.layers:
#                     if l.name.startswith('yolo_output'):
#                         l.set_weights(init_model.get_layer(
#                             l.name).get_weights())
#                     else:
#                         freeze_all(l)

#     #optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
#     # Horovod: adjust learning rate based on number of GPUs.
#     optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate * hvd.size())
#     # Horovod: add Horovod DistributedOptimizer.
#     optimizer = hvd.DistributedOptimizer(optimizer)
#     ###############################################

#     loss = [YoloLoss(anchors[mask]) for mask in anchor_masks]

#     if FLAGS.mode == 'eager_tf':
#         # Eager mode is great for debugging
#         # Non eager graph mode is recommended for real training
#         avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
#         avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

#         for epoch in range(1, FLAGS.epochs + 1):
#             for batch, (images, labels) in enumerate(train_dataset):
#                 with tf.GradientTape() as tape:
#                     outputs = model(images, training=True)
#                     regularization_loss = tf.reduce_sum(model.losses)
#                     pred_loss = []
#                     for output, label, loss_fn in zip(outputs, labels, loss):
#                         pred_loss.append(loss_fn(label, output))
#                     total_loss = tf.reduce_sum(pred_loss) + regularization_loss

#                 grads = tape.gradient(total_loss, model.trainable_variables)
#                 optimizer.apply_gradients(
#                     zip(grads, model.trainable_variables))

#                 logging.info("{}_train_{}, {}, {}".format(
#                     epoch, batch, total_loss.numpy(),
#                     list(map(lambda x: np.sum(x.numpy()), pred_loss))))
#                 avg_loss.update_state(total_loss)

#             for batch, (images, labels) in enumerate(val_dataset):
#                 outputs = model(images)
#                 regularization_loss = tf.reduce_sum(model.losses)
#                 pred_loss = []
#                 for output, label, loss_fn in zip(outputs, labels, loss):
#                     pred_loss.append(loss_fn(label, output))
#                 total_loss = tf.reduce_sum(pred_loss) + regularization_loss

#                 logging.info("{}_val_{}, {}, {}".format(
#                     epoch, batch, total_loss.numpy(),
#                     list(map(lambda x: np.sum(x.numpy()), pred_loss))))
#                 avg_val_loss.update_state(total_loss)

#             logging.info("{}, train: {}, val: {}".format(
#                 epoch,
#                 avg_loss.result().numpy(),
#                 avg_val_loss.result().numpy()))

#             avg_loss.reset_states()
#             avg_val_loss.reset_states()
#             model.save_weights(
#                 'checkpoints/yolov3_train_{}.tf'.format(epoch))
#     else:
#         model.compile(optimizer=optimizer, loss=loss,
#                        run_eagerly=(FLAGS.mode == 'eager_fit'))
       
#         # callbacks = [
#         #     ReduceLROnPlateau(verbose=1),
#         #     EarlyStopping(patience=3, verbose=1),
#         #     ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',
#         #                     verbose=1, save_weights_only=True),
#         #     TensorBoard(log_dir='logs')
#         # ]
#         callbacks = [
                    
#                     # Horovod: broadcast initial variable states from rank 0 to all other processes.
#                     # This is necessary to ensure consistent initialization of all workers when
#                     # training is started with random weights or restored from a checkpoint.
#                     hvd.callbacks.BroadcastGlobalVariablesCallback(0),

#                     # Horovod: average metrics among workers at the end of every epoch.
                    
#                     # Note: This callback must be in the list before the ReduceLROnPlateau,
#                     # TensorBoard or other metrics-based callbacks.
#                     hvd.callbacks.MetricAverageCallback(),

#                     # Horovod: using `lr = 1.0 * hvd.size()` from the very beginning leads to worse final
#                     # accuracy. Scale the learning rate `lr = 1.0` ---> `lr = 1.0 * hvd.size()` during
#                     # the first three epochs. See https://arxiv.org/abs/1706.02677 for details.
#                     hvd.callbacks.LearningRateWarmupCallback(warmup_epochs=3, verbose=1),
#         ]
#         # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
#         if hvd.rank() == 0:
#             callbacks.append(tf.keras.callbacks.ModelCheckpoint('checkpoints/yolov3_train_{epoch}.tf',verbose=1, save_weights_only=True))
        
#         verbose = 1 if hvd.rank() == 0 else 0
        
#         history = model.fit(train_dataset,
#                             steps_per_epoch=5717//FLAGS.batch_size// hvd.size(),
#                             #steps_per_epoch=500,
#                             epochs=FLAGS.epochs,
#                             callbacks=callbacks,
#                             validation_data=val_dataset,
#                             verbose=verbose
#                             )


# if __name__ == '__main__':
#     try:
#         app.run(main)
#     except SystemExit:
#         pass
