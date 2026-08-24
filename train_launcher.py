"""
PE-MMNet v4 统一训练启动器

功能：
1. 单次训练 - 预设模板 + 交互式配置
2. 批量训练 - 队列配置向导 (B+C 流程)
3. 快捷训练 - 快速批量添加命令

使用方式：
    python train_launcher.py               # 交互式（统一入口）
    python train_launcher.py --quick ...   # 快捷批量训练
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# 从根目录导入 launcher（单次训练逻辑）
from launcher import (
    PRESETS, SEGMENTATION_PRESETS, MULTITASK_PRESETS, ABLATION_PRESETS,
    interactive_mode, run_ablation_menu, build_command, save_config,
    queue_config_wizard, load_command_config
)

# 导入批量训练模块
from tools.batch_train_gui import InteractiveCLI, TrainingCommand


def print_banner():
    print("=" * 70)
    print("  PE-MMNet v4 统一训练启动器")
    print("=" * 70)
    print()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='PE-MMNet v4 统一训练启动器')
    parser.add_argument('--quick', nargs='+', help='快捷批量训练: "backbone_2d,epochs,offset,task"')
    args = parser.parse_args()

    print_banner()

    # 快捷模式：直接开始批量训练
    if args.quick:
        cli = InteractiveCLI()
        for quick_str in args.quick:
            cmd = TrainingCommand.from_quick_format(quick_str)
            cli.trainer.add_command(cmd)
        print(f"已添加 {len(args.quick)} 条命令")
        cli._start_training()
        return

    # 交互式模式：统一入口
    cli = InteractiveCLI()

    while True:
        # 显示主菜单
        print("\n" + "=" * 70)
        print("  PE-MMNet v4 训练启动器")
        print("=" * 70)
        print()
        print("  【单次训练】")
        print("    [1] 快速训练（预设模板）")
        print("    [2] 自定义配置（详细参数说明）")
        print("    [3] 消融实验")
        print()
        print("  【批量训练】")
        print("    [4] 队列配置向导 ⭐ 逐步配置，自动加入队列")
        print("    [5] 添加到现有队列")
        print()
        print(f"  当前队列: {len(cli.trainer.commands)} 条命令")
        print()

        if len(cli.trainer.commands) > 0:
            print("  [r] ▶ 开始批量训练")
            print("  [v] 查看队列")
            print("  [c] 清空队列")
        print("  [l] 加载配置")
        print("  [s] 保存配置")
        print("  [q] 退出")
        print()

        choice = input("请选择: ").strip().lower()

        if choice == "q":
            print("\n已退出")
            return

        # ===== 单次训练 =====
        elif choice == "1":
            # 快速训练 - 预设模板
            print("\n【预设模板】\n")
            for key, preset in PRESETS.items():
                print(f"  [{key}] {preset['name']}")
                print(f"       {preset['description']}")
            print("\n  [s] 分割任务  [m] 多任务  [q] 返回")
            print()

            preset_choice = input("请选择: ").strip().lower()

            if preset_choice == "q":
                continue
            elif preset_choice == "s":
                print("\n【分割任务预设】")
                for key, preset in SEGMENTATION_PRESETS.items():
                    print(f"  [{key}] {preset['name']}")
                seg_choice = input("选择: ").strip().lower()
                if seg_choice in SEGMENTATION_PRESETS:
                    preset = SEGMENTATION_PRESETS[seg_choice]
                    args_list = preset['args'].split()
                else:
                    continue
            elif preset_choice == "m":
                print("\n【多任务预设】")
                for key, preset in MULTITASK_PRESETS.items():
                    print(f"  [{key}] {preset['name']}")
                mt_choice = input("选择: ").strip().lower()
                if mt_choice in MULTITASK_PRESETS:
                    preset = MULTITASK_PRESETS[mt_choice]
                    args_list = preset['args'].split()
                else:
                    continue
            elif preset_choice in PRESETS:
                preset = PRESETS[preset_choice]
                args_list = preset['args'].split()
            else:
                continue

            # 构建命令并询问操作
            cmd = build_command(args_list)
            print(f"\n命令: {cmd}\n")
            print("  [1] 立即执行  [2] 加入队列  [q] 返回")
            action = input("请选择: ").strip().lower()

            if action == "q":
                continue
            elif action == "1":
                os.system(cmd)
                return
            elif action == "2":
                tc = TrainingCommand.from_string(cmd)
                success, msg, _ = cli.trainer.add_command(tc)
                if success:
                    print(f"✅ 已添加到队列 (当前 {len(cli.trainer.commands)} 条)")
                else:
                    print(f"⚠️ {msg}")
            continue

        elif choice == "2":
            # 自定义配置
            args_list = interactive_mode()
            cmd = build_command(args_list)

            print(f"\n命令: {cmd}\n")
            print("  [1] 立即执行  [2] 加入队列  [q] 返回")
            action = input("请选择: ").strip().lower()

            if action == "q":
                continue
            elif action == "1":
                os.system(cmd)
                return
            elif action == "2":
                tc = TrainingCommand.from_string(cmd)
                success, msg, _ = cli.trainer.add_command(tc)
                if success:
                    print(f"✅ 已添加到队列 (当前 {len(cli.trainer.commands)} 条)")
                else:
                    print(f"⚠️ {msg}")
            continue

        elif choice == "3":
            # 消融实验
            print("\n【消融实验预设】\n")
            for key, preset in ABLATION_PRESETS.items():
                print(f"  [{key}] {preset['name']}")
            print("\n  [all] 运行全部  [q] 返回")
            print()

            choice = input("选择: ").strip().lower()
            if choice == 'q':
                continue
            if choice == 'all':
                for key, preset in ABLATION_PRESETS.items():
                    if key != 'all':
                        cmd = build_command(preset['args'].split(), is_ablation=True)
                        print(f"\n执行: {preset['name']}")
                        os.system(cmd)
                continue
            elif choice in ABLATION_PRESETS:
                preset = ABLATION_PRESETS[choice]
                cmd = build_command(preset['args'].split(), is_ablation=True)
                print(f"\n执行: {preset['name']}")
                os.system(cmd)
            continue

        # ===== 批量训练 =====
        elif choice == "4":
            # 队列配置向导 (B+C 流程)
            result = queue_config_wizard(cli)
            if result == "start_training":
                print("\n开始批量训练...\n")
                cli._start_training()
            continue

        elif choice == "5":
            # 添加到现有队列（调用批量训练的丰富配置界面）
            cli._add_command()
            continue

        elif choice == "r" and len(cli.trainer.commands) > 0:
            # 开始批量训练
            print("\n开始批量训练...\n")
            cli._start_training()
            continue

        elif choice == "v" and len(cli.trainer.commands) > 0:
            # 查看队列
            cli._show_commands()
            continue

        elif choice == "c" and len(cli.trainer.commands) > 0:
            # 清空队列
            confirm = input("确认清空队列? (y/N): ").strip().lower()
            if confirm == 'y':
                cli.trainer.clear_commands()
                print("队列已清空")
            continue

        elif choice == "l":
            # 加载配置
            config_list = load_command_config()
            if config_list:
                print(f"\n已加载 {len(config_list)} 条配置")
                for args_list in config_list:
                    cmd = build_command(args_list)
                    tc = TrainingCommand.from_string(cmd)
                    success, msg, _ = cli.trainer.add_command(tc, allow_duplicate=True)
                    if success:
                        print(f"  ✅ 已添加: {tc.get_display_name()}")
                    else:
                        print(f"  ⚠️ 已存在: {tc.get_display_name()}")
                print(f"\n当前队列: {len(cli.trainer.commands)} 条命令")
            else:
                print("未找到保存的配置或加载失败")
            continue

        elif choice == "s":
            # 保存配置
            if len(cli.trainer.commands) > 0:
                config_list = []
                for cmd in cli.trainer.commands:
                    # 从 TrainingCommand 转换为参数列表
                    args_list = []
                    for key, value in [
                        ('--variant', cmd.variant),
                        ('--backbone_2d', cmd.backbone_2d),
                        ('--backbone_1d', cmd.backbone_1d),
                        ('--fusion', cmd.fusion),
                        ('--epochs', str(cmd.epochs)),
                        ('--predict_offset', str(cmd.predict_offset)),
                        ('--task', cmd.task),
                        ('--image_size', str(cmd.image_size)),
                        ('--lr', str(cmd.lr)),
                        ('--dropout', str(cmd.dropout)),
                    ]:
                        if key == '--variant' and value == 'resnet18':
                            continue  # 跳过默认值
                        args_list.extend([key, value])
                    config_list.append(args_list)
                save_config(config_list)
                print(f"已保存 {len(config_list)} 条配置到 config_launcher.json")
            else:
                print("队列为空，无需保存")
            continue

        else:
            print("无效选择\n")


if __name__ == "__main__":
    main()
