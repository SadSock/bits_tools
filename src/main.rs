#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release

use eframe::egui;
use std::mem;

fn main() -> Result<(), eframe::Error> {
    env_logger::init(); // Log to stderr (if you run with `RUST_LOG=debug`).
    let options = eframe::NativeOptions {
        initial_window_size: Some(egui::vec2(900.0, 500.0)),
        ..Default::default()
    };
    eframe::run_native(
        "Bits Tool",
        options,
        Box::new(|_cc| Box::<MyApp>::default()),
    )
}

#[derive(PartialEq, Eq)]
enum Panel {
    Bits,
    AMD,
}

#[derive(Clone, PartialEq)]
struct Operator {
    bits: [bool; 32],
    unsign: u32,
    sign: i32,
    float: f32,
    hex_str: String,
    unsign_str: String,
    sign_str: String,
    float_str: String,
    name: String,
}

// impl Default for Operator {
//     fn default() -> Self {
//         Self {
//             bits: [false; 32],
//             unsign: 0,
//             sign: 0,
//             float: 0.0,
//             hex_str: "0x0".to_owned(),
//             unsign_str: 0.to_string(),
//             sign_str: 0.to_string(),
//             float_str: 0.0.to_string(),
//             name: _name,
//         }
//     }
// }

#[derive(Clone, PartialEq)]
struct Bits {
    src0: Operator,
    src1: Operator,
    src2: Operator,
    dest: Operator,
    selected: usize,
}

impl Bits {
    fn draw_ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        window_operator(ctx, ui, &mut self.src0);
        window_operator(ctx, ui, &mut self.src1);
        window_operator(ctx, ui, &mut self.src2);
        let instructions = ["mul_u32", "mul_i32", "add_u32", "add_i32", "fma_f32"];
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
            "fma_f32" => {
                let a = self.src0.float;
                let b = self.src1.float;
                let c = self.src2.float;
                let d = a.mul_add(b, c);
                unsafe {
                    self.dest.unsign = mem::transmute(d);
                }
            }
            _ => {}
        }

        window_operator(ctx, ui, &mut self.dest);
        update_operator(&mut self.src1);
        update_operator(&mut self.src0);
        update_operator(&mut self.src2);
        update_operator(&mut self.dest);
    }
}

impl Default for Bits {
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
                name: "src0".to_owned(),
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
                name: "src1".to_owned(),
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
                name: "src2".to_owned(),
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
                name: "dest".to_owned(),
            },
            selected: 0,
        }
    }
}

struct MyApp {
    open_panel: Panel,
    x86_panel: Bits,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            open_panel: Panel::Bits,
            x86_panel: Bits::default(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.separator();
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.open_panel, Panel::AMD, "AMD");
                ui.selectable_value(&mut self.open_panel, Panel::Bits, "X86");
            });
            ui.separator();

            match self.open_panel {
                Panel::Bits => {
                    self.x86_panel.draw_ui(ctx, ui);
                }

                Panel::AMD => {}
            }
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

fn window_operator(ctx: &egui::Context, ui: &mut egui::Ui, op: &mut Operator) {
    ui.collapsing(op.name.to_owned(), |ui| {
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
    });

    ui.horizontal(|ui| {
        let response = ui.add(egui::TextEdit::singleline(&mut op.hex_str).desired_width(100.0));
        if response.changed() {
            // …
        }

        let response = ui.add(egui::TextEdit::singleline(&mut op.unsign_str).desired_width(100.0));
        if response.changed() {
            // …
        }
        let response = ui.add(egui::TextEdit::singleline(&mut op.sign_str).desired_width(100.0));
        if response.changed() {
            // …
        }
        let response = ui.add(egui::TextEdit::singleline(&mut op.float_str).desired_width(350.0));
        if response.changed() {
            // …
        }
    });
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
}
