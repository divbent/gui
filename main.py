from time import sleep
from PIL import Image
from ultralytics import YOLO

model = YOLO('best.pt')
# model.predict('2.jpg', save=True, show=True, show_labels=False, project='/home/a/Desktop')
# model.predict('2.jpg', show=True, show_labels=False)
# sleep(20)

results = model(['2.jpg'])  # results list

# Visualize the results
for i, r in enumerate(results):
    # Plot results image
    im_bgr = r.plot()  # BGR-order numpy array
    im_rgb = Image.fromarray(im_bgr[..., ::-1])
    r.show(conf=False, line_width=5, labels=False)