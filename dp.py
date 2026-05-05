# dynamic programming adds a soothness constrain , therefore it increases accuracy
# the idea is to treat each horizontal scanline of the left and right images as a sequence and find the optimal path that aligns them
import numpy as np
import cv2
def disparityDP(L, R, window_size=5, max_disparity=64, smoothness_penalty=1, occlusion_penalty=2):
    # D[0][0]=0
    height, width = L.shape
    disparity_map = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        #cost matrix for row columns: pixel position (x), rows = possible disparity values (d).
        D = np.zeros((width, max_disparity))
        # start with an occulsion penalty for the first pixel in the row
        for x in range(width):
            D[x][0] = x * occlusion_penalty
        for d in range(max_disparity):
            D[0][d] = d * occlusion_penalty
        for x in range(1, width):
            for d in range(max_disparity):
                # calculating data cost calculating the diff bet the 2 pixels
                if x - d >= 0:
                    match_cost = abs(int(L[y, x]) - int(R[y, x - d]))
                else:
                    match_cost = 255 # Forbidden
                    
                # Transitions from previous pixel (x-1)
                # Option 1: Stay at same disparity
                c1 = D[x-1, d]
                
                # Option 2: Transition from d-1 (Smoothness penalty)
                c2 = D[x-1, d-1] + smoothness_penalty if d > 0 else float('inf')
                
                # Option 3: Transition from d+1 (Smoothness penalty)
                c3 = D[x-1, d+1] + smoothness_penalty if d < max_disparity-1 else float('inf')
                
                # 3. Store result
                D[x, d] = match_cost + min(c1, c2, c3)

        # find the index of the lowest acc cost
        current_d = np.argmin(D[width - 1, :])
            
        for x in range(width - 1, -1, -1):
            disparity_map[y, x] = current_d
            
            # Decide which d to use for the next (previous) pixel x-1
            if x > 0:
                # Look at the three possible parents in the previous column
                prev_costs = [
                    D[x-1, current_d], # c1: stay
                    D[x-1, current_d-1] if current_d > 0 else float('inf'), # c2: up
                    D[x-1, current_d+1] if current_d < max_disparity-1 else float('inf') # c3: down
                ]
                # Update current_d based on which parent was the minimum
                best_parent_idx = np.argmin(prev_costs)
                if best_parent_idx == 1:
                    current_d -= 1
                elif best_parent_idx == 2:
                    current_d += 1
    return disparity_map

if __name__ == "__main__":
    L = cv2.imread(r'materials\l1.png', cv2.IMREAD_GRAYSCALE)
    R = cv2.imread(r'materials\r1.png', cv2.IMREAD_GRAYSCALE)
    disparity_map = disparityDP(L, R, window_size=5, max_disparity=64, smoothness_penalty=5, occlusion_penalty=2)
    vis_map = cv2.normalize(disparity_map, None, 0, 255, cv2.NORM_MINMAX)
    color_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)

    # Save the results
    cv2.imwrite('disparity_dp_gray.png', vis_map)
    cv2.imwrite('disparity_dp_color.png', color_map)
    
    print("Success! Saved 'disparity_dp_gray.png' and 'disparity_dp_color.png'.")