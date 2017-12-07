import net.utility.draw  as nud

def save_topview_batch(batch_top_view, name):

    for i in range(batch_top_view.shape[-1]):
        img = batch_top_view[0, :, :, i].reshape(batch_top_view.shape[1], batch_top_view.shape[2])
        nud.imsave('TopView_' + name.split('/')[-1] + str(i), img)

def save_rgb_batch(batch_rgb_images, batch_gt_boxes3d, name):

    rgb = nud.draw_box3d_on_camera(batch_rgb_images[0], batch_gt_boxes3d[0])
    nud.imsave('RGB_' + name.split('/')[-1], rgb)