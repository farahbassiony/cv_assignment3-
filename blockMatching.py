import numpy as np
import cv2
import numpy as np
import cv2

def disparitySSD(L, R, window_size=5, max_disparity=64):
    height, width = L.shape
    disparity_map = np.zeros((height, width), dtype=np.uint8)

    half_w = window_size // 2

    for y in range(half_w, height - half_w):
        for x in range(half_w, width - half_w):

            best_ssd = float('inf')
            best_d = 0

            for d in range(max_disparity):

                if x - d - half_w < 0:
                    continue

                ssd = 0

                for i in range(-half_w, half_w + 1):
                    for j in range(-half_w, half_w + 1):

                        diff = int(L[y+i, x+j]) - int(R[y+i, x+j-d])
                        ssd += diff * diff   # <-- key difference from SAD

                if ssd < best_ssd:
                    best_ssd = ssd
                    best_d = d

            disparity_map[y, x] = best_d

    return disparity_map

def disparitySAD(L,R,window_size,max_disparity=64):
    height, width = L.shape
    disparity_map = np.zeros((height, width), dtype=np.uint8)

    half_w = window_size // 2

    for y in range(half_w, height - half_w):
        for x in range(half_w, width - half_w):
            best_sad = float('inf')
            best_d = 0
            for d in range(max_disparity):
                #keeps the search in range
                if x - d - half_w < 0:
                    continue
                sad = 0
                # loop over windows
                for i in range(-half_w, half_w + 1):
                    #loop  over columns
                    for j in range(-half_w, half_w + 1):
                        # take the pixel from the left with its corresponding right pixel
                        sad += abs(int(L[y+i, x+j]) - int(R[y+i, x+j-d]))
                if sad < best_sad:
                    best_sad = sad
                    best_d = d
            disparity_map[y, x] = best_d
    return disparity_map

if __name__ == "__main__":

    image_pairs = [
        ("materials/l1.png", "materials/r1.png"),
        ("materials/l2.png", "materials/r2.png"),
        ("materials/l3.png", "materials/r3.png")
    ]

    pair=1
    for left_path, right_path in image_pairs:

        imgL = cv2.imread(left_path, 0)
        imgR = cv2.imread(right_path, 0)

        count=1
        for w in [1,5,9]:
            disp = disparitySAD(imgL, imgR, w, max_disparity=64)

            disp_vis = (disp / disp.max() * 255).astype(np.uint8)
            # cv2.imshow(f"Disparity {count}pair{pair}", disp_vis)
            cv2.imwrite(f"SAD{count}pair{pair}.png", disp_vis)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            count += 1
        count = 1
        for w in [1,5,9]:
            disp = disparitySSD(imgL, imgR, w, max_disparity=64)

            disp_vis = (disp / disp.max() * 255).astype(np.uint8)
            cv2.imshow(f"Disparity {count}pair{pair}", disp_vis)
            cv2.imwrite(f"SSD{count}pair{pair}.png", disp_vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            count += 1  
        pair+=1