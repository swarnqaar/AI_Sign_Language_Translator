# Sign language gesture class labels (ASL A-Z + common phrases)
SIGN_CLASSES = [
    "A","B","C","D","E","F","G","H","I","J",
    "K","L","M","N","O","P","Q","R","S","T",
    "U","V","W","X","Y","Z",
    "Hello","Thank You","Yes","No","Please","Sorry",
    "Help","More","Done","Good",
]
NUM_CLASSES = len(SIGN_CLASSES)

# MediaPipe: 21 landmarks x 3 coords (x,y,z) x 2 hands = 126 features
NUM_LANDMARKS = 21
NUM_COORDS_PER_LANDMARK = 3
NUM_HANDS = 2
FEATURE_SIZE = NUM_LANDMARKS * NUM_COORDS_PER_LANDMARK * NUM_HANDS  # 126
