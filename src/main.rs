#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use std::mem;

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(900.0, 600.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Bits Tool",
        options,
        Box::new(|_cc| Box::<MyApp>::default()),
    )
}

struct Operator {
    bits: [bool; 32],
    unsign: u32,
    sign: i32,
    float: f32,
    hex_str: String,
    unsign_str: String,
    sign_str: String,
    float_str: String,
}

struct MyApp {
    src0: Operator,
    src1: Operator,
    src2: Operator,
    dest: Operator,
    selected: usize,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            src0: Operator {
                bits: [false; 32],
                unsign: 0,
                sign: 0,
                float: 0.0,
                hex_str: "0x0".to_owned(),
                unsign_str: 0.to_string(),
                sign_str: 0.to_string(),
                float_str: 0.0.to_string(),
            },
            src1: Operator {
                bits: [false; 32],
                unsign: 0,
                sign: 0,
                float: 0.0,
                hex_str: "0x0".to_owned(),
                unsign_str: 0.to_string(),
                sign_str: 0.to_string(),
                float_str: 0.0.to_string(),
            },
            src2: Operator {
                bits: [false; 32],
                unsign: 0,
                sign: 0,
                float: 0.0,
                hex_str: "0x0".to_owned(),
                unsign_str: 0.to_string(),
                sign_str: 0.to_string(),
                float_str: 0.0.to_string(),
            },
            dest: Operator {
                bits: [false; 32],
                unsign: 0,
                sign: 0,
                float: 0.0,
                hex_str: "0x0".to_owned(),
                unsign_str: 0.to_string(),
                sign_str: 0.to_string(),
                float_str: 0.0.to_string(),
            },
            selected: 0,
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            window_operator(ctx, _frame, ui, &mut self.src0);
            window_operator(ctx, _frame, ui, &mut self.src1);
            window_operator(ctx, _frame, ui, &mut self.src2);
            let instructions = ["mul_u32", "mul_i32", "add_u32", "add_i32"];
            egui::ComboBox::from_label("Select Instruction!").show_index(
                ui,
                &mut self.selected,
                instructions.len(),
                |i| instructions[i],
            );

            match instructions[self.selected] {
                "mul_u32" => {
                    let a = self.src0.unsign;
                    let b = self.src1.unsign;
                    let c = a.wrapping_mul(b);
                    self.dest.unsign = c;
                }
                "mul_i32" => {
                    let a = self.src0.sign;
                    let b = self.src1.sign;
                    let c = a.wrapping_mul(b);
                    unsafe {
                        self.dest.unsign = mem::transmute(c);
                    }
                }
                "add_u32" => {
                    let a = self.src0.unsign;
                    let b = self.src1.unsign;
                    let c = a.wrapping_add(b);
                    self.dest.unsign = c;
                }
                "add_i32" => {
                    let a = self.src0.sign;
                    let b = self.src1.sign;
                    let c = a.wrapping_add(b);
                    unsafe {
                        self.dest.unsign = mem::transmute(c);
                    }
                }
                _ => {}
            }

            window_operator(ctx, _frame, ui, &mut self.dest);
            update_operator(&mut self.src1);
            update_operator(&mut self.src0);
            update_operator(&mut self.src2);
            update_operator(&mut self.dest);
        });
    }
}

fn update_operator(op: &mut Operator) {
    unsafe {
        op.sign = mem::transmute(op.unsign);
        op.float = mem::transmute(op.unsign);
    }
    op.hex_str = format!("0x{:X}", op.unsign);
    op.sign_str = format!("{}", op.sign);
    op.unsign_str = format!("{}", op.unsign);
    op.float_str = format!("{}", op.float);
    for i in 0..32 {
        if op.unsign & (0x1 << i) != 0 {
            op.bits[i] = true;
        } else {
            op.bits[i] = false;
        }
    }
}

fn window_operator(
    ctx: &egui::Context,
    frame: &mut eframe::Frame,
    ui: &mut egui::Ui,
    op: &mut Operator,
) {
    ui.group(|ui| {
        ui.horizontal(|ui| {
            // signed bit
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.label("sign");
                    for i in 0..1 {
                        ui.vertical(|ui| {
                            if ui.checkbox(&mut op.bits[31 - i], "").clicked() {
                                if op.bits[31 - i] {
                                    op.unsign = op.unsign | (0x1 << (31 - i));
                                } else {
                                    op.unsign = op.unsign & !(0x1 << (31 - i));
                                }
                            }
                            ui.label(format!("{}", 31 - i));
                        });
                    }
                })
            });

            // exp bits
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.label("exp");
                    ui.horizontal(|ui| {
                        for i in 1..9 {
                            ui.vertical(|ui| {
                                if ui.checkbox(&mut op.bits[31 - i], "").clicked() {
                                    if op.bits[31 - i] {
                                        op.unsign = op.unsign | (0x1 << (31 - i));
                                    } else {
                                        op.unsign = op.unsign & !(0x1 << (31 - i));
                                    }
                                }
                                ui.label(format!("{}", 31 - i));
                            });
                        }
                    });
                });
            });

            // exp mantissa
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.label("mantissa");
                    ui.horizontal(|ui| {
                        for i in 9..32 {
                            ui.vertical(|ui| {
                                if ui.checkbox(&mut op.bits[31 - i], "").clicked() {
                                    if op.bits[31 - i] {
                                        op.unsign = op.unsign | (0x1 << (31 - i));
                                    } else {
                                        op.unsign = op.unsign & !(0x1 << (31 - i));
                                    }
                                }
                                ui.label(format!("{}", 31 - i));
                            });
                        }
                    });
                });
            });
        });

        if ui.text_edit_singleline(&mut op.hex_str).changed() {}
        if ui.text_edit_singleline(&mut op.unsign_str).changed() {}
        if ui.text_edit_singleline(&mut op.sign_str).changed() {}
        if ui.text_edit_singleline(&mut op.float_str).changed() {}

        ui.horizontal(|ui| {
            if ui.button("-1").clicked() {
                unsafe {
                    op.unsign = mem::transmute(-1);
                }
            }
            if ui.button("0").clicked() {
                op.unsign = 0;
            }

            if ui.button("des").clicked() {
                if op.unsign == 0 {
                    op.unsign = u32::MAX;
                } else {
                    op.unsign = op.unsign - 1;
                }
            }
            if ui.button("inc").clicked() {
                if op.unsign == u32::MAX {
                    op.unsign = 0;
                } else {
                    op.unsign = op.unsign + 1;
                }
            }

            if ui.button("lshr").clicked() {
                op.unsign = op.unsign.checked_shr(1).unwrap();
            }

            if ui.button("lshl").clicked() {
                op.unsign = op.unsign.checked_shl(1).unwrap();
            }

            if ui.button("ashr").clicked() {
                unsafe {
                    let mut tmp: i32 = mem::transmute(op.unsign);
                    tmp = tmp.checked_shr(1).unwrap();
                    op.unsign = mem::transmute(tmp);
                }
            }
        });
    });
}
