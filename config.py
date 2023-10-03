# training
batch_size = 32
learning_rate = 0.0005
num_epochs = 50
training_part = 0.8  # part of the images that will be used for training rather than testing
weights_folder = "weights/"

save_weights = True
use_pretrained_weights = False
use_pretrained_backbone = True

# interim results
print_batch_after_epoch = True
print_batch_folder = 'out/temp_imgs/'
print_batch_num = 5

img_size = 256

# loss branch activation
ACTIVATE_WORD_FG_BCE = False
ACTIVATE_WORD_FG_DICE = False
ACTIVATE_WORD_TBLR = False
ACTIVATE_WORD_ORIENT = True
ACTIVATE_CHAR_FG_BCE = False
ACTIVATE_CHAR_FG_DICE = False
ACTIVATE_CHAR_TBLR = False
ACTIVATE_CHAR_ORIENT = False

