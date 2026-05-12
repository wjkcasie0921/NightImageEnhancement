import 'dart:typed_data';

import 'package:flutter/material.dart';

/// 前后对比滑块
/// 需求：
/// 1. 中间一条白色竖线作为分割指示器
/// 2. thumb 更明显
class BeforeAfterSlider extends StatefulWidget {
  final Uint8List? before;
  final Uint8List? after;

  const BeforeAfterSlider({
    super.key,
    this.before,
    this.after,
  });

  @override
  State<BeforeAfterSlider> createState() => _BeforeAfterSliderState();
}

class _BeforeAfterSliderState extends State<BeforeAfterSlider> {
  double _value = 0.5;

  @override
  Widget build(BuildContext context) {
    if (widget.before == null || widget.after == null) {
      return const Center(
        child: Text('请选择图片并完成增强'),
      );
    }

    return Column(
      children: [
        Expanded(
          child: LayoutBuilder(
            builder: (context, constraints) {
              return Stack(
                children: [
                  Positioned.fill(
                    child: Image.memory(
                      widget.after!,
                      fit: BoxFit.cover,
                    ),
                  ),
                  Positioned.fill(
                    child: ClipRect(
                      child: Align(
                        alignment: Alignment.centerLeft,
                        widthFactor: _value,
                        child: Image.memory(
                          widget.before!,
                          fit: BoxFit.cover,
                          width: constraints.maxWidth,
                        ),
                      ),
                    ),
                  ),

                  // 白色竖线：让用户更容易感知分割位置
                  Positioned(
                    left: constraints.maxWidth * _value - 1,
                    top: 0,
                    bottom: 0,
                    child: Container(
                      width: 2,
                      color: Colors.white,
                    ),
                  ),
                ],
              );
            },
          ),
        ),
        SliderTheme(
          data: SliderTheme.of(context).copyWith(
            trackHeight: 3,
            activeTrackColor: Colors.white,
            inactiveTrackColor: Colors.white24,
            thumbColor: Colors.white,
            overlayColor: Colors.white24,
            thumbShape: const RoundSliderThumbShape(
              enabledThumbRadius: 12,
            ),
            overlayShape: const RoundSliderOverlayShape(
              overlayRadius: 22,
            ),
          ),
          child: Slider(
            value: _value,
            min: 0.0,
            max: 1.0,
            onChanged: (v) {
              setState(() => _value = v);
            },
          ),
        ),
      ],
    );
  }
}
