import 'dart:typed_data';

import 'package:image/image.dart' as img;

/// 后处理
/// 支持：
/// 1. 单输出增强图：直接 tensor -> JPEG
/// 2. 双输出 Retinex：S = R ◦ L 合成后再输出
class ImagePostprocessor {
  /// 单输出模式：tensor -> image
  Uint8List postprocessSingleOutput(dynamic enhancedTensor, int size) {
    final output = img.Image(width: size, height: size);

    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        final r = (enhancedTensor[0][y][x][0] * 255.0).clamp(0, 255).toInt();
        final g = (enhancedTensor[0][y][x][1] * 255.0).clamp(0, 255).toInt();
        final b = (enhancedTensor[0][y][x][2] * 255.0).clamp(0, 255).toInt();

        // image 4.x 推荐使用 ColorRgb8
        output.setPixel(x, y, img.ColorRgb8(r, g, b));
      }
    }

    return Uint8List.fromList(img.encodeJpg(output, quality: 95));
  }

  /// 双输出模式：refl + illum -> 合成增强图
  ///
  /// S = R ◦ L
  /// 其中：
  /// - refl: [1, H, W, 3]
  /// - illum: [1, H, W, 1]
  Uint8List postprocessRetinex({
    required dynamic reflTensor,
    required dynamic illumTensor,
    required int size,
  }) {
    final output = img.Image(width: size, height: size);

    for (int y = 0; y < size; y++) {
      for (int x = 0; x < size; x++) {
        final illum = illumTensor[0][y][x][0].toDouble();

        final r = (reflTensor[0][y][x][0] * illum * 255.0).clamp(0, 255).toInt();
        final g = (reflTensor[0][y][x][1] * illum * 255.0).clamp(0, 255).toInt();
        final b = (reflTensor[0][y][x][2] * illum * 255.0).clamp(0, 255).toInt();

        output.setPixel(x, y, img.ColorRgb8(r, g, b));
      }
    }

    // 输出 JPEG，quality=95，兼顾体积和视觉质量
    return Uint8List.fromList(img.encodeJpg(output, quality: 95));
  }
}
