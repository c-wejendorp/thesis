NOISE_FOLDER = "_background_noise_"
NOISE_FILES = [
    "doing_the_dishes.wav",
    "dude_miaowing.wav",
    "exercise_bike.wav",
    "pink_noise.wav",
    "running_tap.wav",
    "white_noise.wav",
]
SAMPLE_RATE = 16000
TARGET_LENGTH = 16000  # 1 second

ALL_KEYWORDS = ["backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
            "follow", "forward", "four", "go", "happy", "house", "learn", "left",
            "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
            "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero"]

CANONICAL = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

COMMAND_KEYWORDS_20 = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                    "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]


COMMAND_KEYWORDS_24 = COMMAND_KEYWORDS_20 + ["backward", "forward", "follow", "learn"]