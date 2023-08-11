# training
batch_size = 32
learning_rate = 0.0005
num_epochs = 20
training_part = 0.75  # part of the images that will be used for training rather than testing

save_weights = False

# interim results
print_batch_after_epoch = True
print_batch_folder = 'out/temp_imgs/'
print_batch_num = 5

img_size = 256

# loss branch activation
ACTIVATE_WORD_FG_BCE = True
ACTIVATE_WORD_FG_DICE = True
ACTIVATE_WORD_TBLR = False
ACTIVATE_WORD_ORIENT = False
ACTIVATE_CHAR_FG_BCE = False
ACTIVATE_CHAR_FG_DICE = False
ACTIVATE_CHAR_TBLR = False
ACTIVATE_CHAR_ORIENT = False

