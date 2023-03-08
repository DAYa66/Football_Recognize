import os
import os.path as osp
import glob
from collections import OrderedDict
import numpy as np
import cv2
from Football_Recognize.utils.images import imwrite, resize_height


class VideoDemo:
    def __init__(self,model,fps=30,save_path=None,buffer_size=0,show_video=True,max_frame_cn=None,interval=None,
         file_pattern="{:06d}.jpg",args=None) -> None:
        self.model = model
        self.fps = fps
        self.save_path = save_path
        self.buffer_size = buffer_size
        self.buffer = []
        self.write_size = None
        self.video_reader = None
        self.video_writer = None
        self.show_video = show_video
        self.preprocess = None
        self.max_frame_cn = max_frame_cn
        self.interval = interval
        self.file_pattern = file_pattern
        self.args = args
        self.track_data = []
    
    def __del__(self):
        self.close()
    
    def close(self):
        if self.video_writer is not None:
            self.video_writer.release()
        if hasattr(self,"model"):
            del self.model

    def init_reader(self):
        if self.video_path is not None and osp.exists(self.video_path):
            print(f"Use video file {self.video_path}")
            self.video_reader = VideoReader(self.video_path,file_pattern=self.file_pattern)
            self.video_writer_path = osp.join(osp.abspath(osp.join(self.video_path, os.pardir)), "results", self.video_path.split(os.sep)[-1])
            print(f"Save video to {self.video_writer_path}")
            self.video_writer = VideoWriter(self.video_writer_path, 50, "BGR")
            self.frame_cnt = self.video_reader.frames_nr
            if self.max_frame_cn is not None and self.max_frame_cn>1:
                self.frame_cnt = min(self.frame_cnt,self.max_frame_cn)
        else:
            if self.video_path is not None:
                vc = int(self.video_path)
            else:
                vc = -1
            print(f"Use camera {vc}")
            self.video_reader = cv2.VideoCapture(vc)
            self.frame_cnt = -1


    def inference_loop(self,video_path=None):
        self.video_path = video_path
        self.init_reader()
        idx = 0

        for frame in self.video_reader:
            idx += 1
            # print(idx)
            if self.interval is not None and self.interval>1:
                if idx%self.interval != 0:
                    continue
            self.model.idx = idx
            if self.preprocess is not None:
                frame = self.preprocess(frame)
            img = self.inference(frame)
            save_path = osp.join(self.save_path,self.file_pattern.format(idx))
            if self.args is None or self.args.log_imgs:
                imwrite(save_path, img)
            if self.video_writer is not None:
                self.video_writer.write(img[..., ::-1])
            if self.show_video:
                cv2.imshow("video",img[...,::-1])
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            if self.frame_cnt > 1:
                # sys.stdout.write(f"{idx}/{self.frame_cnt}  {idx*100/self.frame_cnt:.3f}%.\r")
                if idx>self.frame_cnt:
                    break

    def inference(self,img):
        if self.buffer_size <= 1:
            r_img = self.inference_single_img(img)
        else:
            r_img = self.inference_buffer_img(img)
        return r_img

    def inference_single_img(self,img):
        return self.model(img)

    def inference_buffer_img(self,img):
        self.buffer.append(img)
        if len(self.buffer)>self.buffer_size:
            self.buffer = self.buffer[-self.buffer_size:]
        return self.model(self.buffer)

    @staticmethod
    def get_last_img(imgs):
        img = imgs[-1]
        if isinstance(img,dict):
            if 'raw_image' in img:
                return img['raw_image']
            return img['image']
        else:
            return img

    @staticmethod
    def resize_h_and_save_raw_image_preprocess(img,h=224):
        r_img = resize_height(img,h).astype(np.uint8)
        return {'image':r_img,"raw_image":img}
    

class VideoReader:
    def __init__(self,path,file_pattern="img_{:05d}.jpg",suffix=".jpg",preread_nr=0) -> None:
        if os.path.isdir(path):
            self.dir_path = path
            self.reader = None
            self.all_files = glob.glob(os.path.join(path,"*"+suffix))
            self.frames_nr = len(self.all_files)
            self.fps = 1
        else:
            self.reader = cv2.VideoCapture(path)
            self.dir_path = None
            self.frames_nr = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = self.reader.get(cv2.CAP_PROP_FPS)
            self.preread_nr = preread_nr
            if self.preread_nr>1:
                self.reader_buffer = OrderedDict()
            else:
                self.reader_buffer = None


        self.idx = 1
        self.file_pattern = file_pattern

    def __iter__(self):
        return self
    
    def __getitem__(self,idx):
        if self.dir_path is None:
            if self.preread_nr>1:
                if idx in self.reader_buffer:
                    return self.reader_buffer[idx]
                elif idx<self.idx-1:
                    raise NotImplemented()
                else:
                    for x in range(self.idx-1,idx+1):
                        if x in self.reader_buffer:
                            continue
                        ret,frame = self.reader.read()
                        if ret:
                            frame = frame[...,::-1]
                            self.reader_buffer[x] = frame
                    if idx in self.reader_buffer:
                        return self.reader_buffer[idx]

            raise NotImplemented()
        elif idx<self.frames_nr:
            if self.file_pattern is None:
                file_path = self.all_files[idx]
            else:
                file_path = os.path.join(self.dir_path,self.file_pattern.format(idx+1))
            img = cv2.imread(file_path)
            return img[...,::-1]
        else:
            raise RuntimeError()
    
    def __len__(self):
        if self.dir_path is not None:
            return self.frames_nr
        elif self.reader is not None:
            return self.frames_nr
        else:
            raise RuntimeError()

    def __next__(self):
        """
        return frame in RGB
        """
        if self.reader is not None:
            if self.preread_nr>1:
                if self.idx-1 in self.reader_buffer:
                    frame = self.reader_buffer[self.idx-1]
                    ret = True
                else:
                    ret,frame = self.reader.read()
                    if not ret:
                        raise StopIteration()
                    frame = frame[...,::-1]
                    self.reader_buffer[self.idx-1] = frame
                    while len(self.reader_buffer)>self.preread_nr:
                        self.reader_buffer.popitem(last=False)
            else:
                retry_nr = 10
                while retry_nr>0:
                    ret,frame = self.reader.read()
                    retry_nr -= 1
                    if ret:
                        break

                if ret:
                    frame = frame[...,::-1]
            
            self.idx += 1
            if not ret:
                raise StopIteration()
            else:
                return frame
        else:
            if self.idx>self.frames_nr:
                raise StopIteration()
            if self.file_pattern is not None:
                file_path = osp.join(self.dir_path,self.file_pattern.format(self.idx))
            else:
                file_path = self.all_files[self.idx-1]
            img = cv2.imread(file_path)
            self.idx += 1
            return img[...,::-1]

class VideoWriter:
    def __init__(self,filename,fps=30,fmt='RGB'):
        self.video_writer = None
        self.fmt = fmt
        self.fps = fps
        self.filename = filename

    def __del__(self):
        self.release()

    def init_writer(self,img):
        if self.video_writer is not None:
            return
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        write_size = img.shape[:2][::-1]
        self.video_writer = cv2.VideoWriter(self.filename, fourcc, self.fps, write_size)

    def write(self,img):
        if self.video_writer is None:
            self.init_writer(img)
        fmt = self.fmt
        if fmt == "BGR":
            self.video_writer.write(img)
        elif fmt=="RGB":
            self.video_writer.write(img[...,::-1])
        else:
            print(f"ERROR fmt {fmt}.")

    def release(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
