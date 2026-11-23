import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class MesoNet(nn.Module):
    """MesoNet visual baseline that returns one logit per clip."""

    def __init__(self):
        super(MesoNet, self).__init__()
        self.output_type = 'logits'
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 16, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn4 = nn.BatchNorm2d(16)
        self.pool4 = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(1024, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = F.dropout(F.relu(self.fc1(x)), 0.5, training=self.training)
        return self.fc2(x)

    def process_video(self, frames_batch):
        batch_size, num_frames = frames_batch.shape[:2]
        frames_reshaped = frames_batch.view(-1, *frames_batch.shape[2:])
        if frames_reshaped.shape[-2:] != (256, 256):
            frames_reshaped = F.interpolate(frames_reshaped, size=(256, 256), mode='bilinear', align_corners=False)
        frame_preds = self.forward(frames_reshaped)
        return frame_preds.view(batch_size, num_frames, 1).mean(dim=1)


class XceptionNet(nn.Module):
    """XceptionNet visual baseline that returns one logit per clip."""

    def __init__(self, pretrained=True):
        super(XceptionNet, self).__init__()
        self.output_type = 'logits'
        self.xception_base = timm.create_model('xception', pretrained=pretrained, num_classes=1)

    def forward(self, x):
        return self.xception_base(x)

    def process_video(self, video_frames):
        batch_size, num_frames = video_frames.shape[:2]
        frames_reshaped = video_frames.view(-1, *video_frames.shape[2:])
        if frames_reshaped.shape[-2:] != (299, 299):
            frames_reshaped = F.interpolate(frames_reshaped, size=(299, 299), mode='bilinear', align_corners=False)
        frame_preds = self.forward(frames_reshaped)
        frame_preds = frame_preds.view(batch_size, num_frames, 1)
        return frame_preds.mean(dim=1), frame_preds


class LipSyncDetector(nn.Module):
    """Audio-visual baseline that scores frames separately before temporal averaging."""

    def __init__(self):
        super(LipSyncDetector, self).__init__()
        self.output_type = 'logits'
        self.visual_conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.visual_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.visual_conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.visual_pool = nn.MaxPool2d(2, 2)
        self.audio_conv1 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.audio_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.audio_pool = nn.MaxPool1d(2, 2)
        self.visual_features = 256 * 32 * 32
        self.audio_features = 128 * 32
        self.fusion_fc1 = nn.Linear(self.visual_features + self.audio_features, 512)
        self.fusion_fc2 = nn.Linear(512, 128)
        self.fusion_fc3 = nn.Linear(128, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, visual, audio):
        v = F.relu(self.visual_conv1(visual))
        v = self.visual_pool(v)
        v = F.relu(self.visual_conv2(v))
        v = self.visual_pool(v)
        v = F.relu(self.visual_conv3(v))
        v = self.visual_pool(v)
        v = v.view(v.size(0), -1)
        a = F.relu(self.audio_conv1(audio))
        a = self.audio_pool(a)
        a = F.relu(self.audio_conv2(a))
        a = self.audio_pool(a)
        a = a.view(a.size(0), -1)
        fusion = torch.cat((v, a), dim=1)
        fusion = F.relu(self.fusion_fc1(fusion))
        fusion = self.dropout(fusion)
        fusion = F.relu(self.fusion_fc2(fusion))
        fusion = self.dropout(fusion)
        return self.fusion_fc3(fusion)

    def process_video(self, video_frames, audio_features):
        if video_frames.dim() != 5:
            raise ValueError("video_frames must have shape (batch, frames, channels, height, width)")
        batch_size, num_frames = video_frames.shape[:2]
        visual = video_frames.view(-1, *video_frames.shape[2:])
        if visual.shape[-2:] != (256, 256):
            visual = F.interpolate(visual, size=(256, 256), mode='bilinear', align_corners=False)

        audio = self._prepare_audio(audio_features)
        audio = audio.unsqueeze(1).expand(-1, num_frames, -1, -1)
        audio = audio.reshape(batch_size * num_frames, *audio.shape[2:])
        frame_preds = self.forward(visual, audio)
        predictions = frame_preds.view(batch_size, num_frames, 1).mean(dim=1)
        attention_weights = torch.ones(batch_size, 2, device=predictions.device)
        return predictions, attention_weights

    def _prepare_audio(self, audio_features):
        if audio_features.dim() == 4:
            audio = audio_features.squeeze(1)
        elif audio_features.dim() == 3:
            if audio_features.shape[1] <= audio_features.shape[2]:
                audio = audio_features
            else:
                audio = audio_features.transpose(1, 2)
        elif audio_features.dim() == 2:
            audio = audio_features.unsqueeze(-1)
        else:
            raise ValueError("audio_features must have 2, 3, or 4 dimensions")

        if audio.shape[1] != 128:
            audio = audio.transpose(1, 2)
            audio = F.interpolate(audio, size=128, mode='linear', align_corners=False)
            audio = audio.transpose(1, 2)

        if audio.shape[-1] != 128:
            audio = F.interpolate(audio, size=128, mode='linear', align_corners=False)
        return audio
