#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(float_next_up_down)]
#![feature(unchecked_math)]

use eframe::egui;
use regex::Regex;
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

fn is_hex(s: &str) -> bool {
    let re = Regex::new(r"^0x[0-9a-fA-F]{1,8}$").unwrap();
    re.is_match(s)
}

#[derive(PartialEq, Eq)]
enum Panel {
    Rust,
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

#[derive(Clone, PartialEq)]
struct Rust {
    src0: Operator,
    src1: Operator,
    src2: Operator,
    dest: Operator,
    selected: usize,
}

impl Rust {
    fn draw_ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        window_operator(ctx, ui, &mut self.src0);
        window_operator(ctx, ui, &mut self.src1);
        window_operator(ctx, ui, &mut self.src2);
        let instructions = ["fma_f32", "mul_u32", "mul_i32", "add_u32", "add_i32"];
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

impl Default for Rust {
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

#[derive(Clone, PartialEq)]
struct AMD {
    src0: Operator,
    src1: Operator,
    src2: Operator,
    dest: Operator,
    selected: usize,
}

impl AMD {
    fn draw_ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        window_operator(ctx, ui, &mut self.src0);
        window_operator(ctx, ui, &mut self.src1);
        window_operator(ctx, ui, &mut self.src2);
        let instructions = ["fma_f32", "mul_u32", "mul_i32", "add_u32", "add_i32"];
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

impl Default for AMD {
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
    panel: Panel,
    amd_panel: AMD,
    rust_panel: Rust,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            panel: Panel::Rust,
            amd_panel: AMD::default(),
            rust_panel: Rust::default(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.separator();
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.panel, Panel::AMD, "AMD");
                ui.selectable_value(&mut self.panel, Panel::Rust, "Rust");
            });
            ui.separator();

            match self.panel {
                Panel::Rust => {
                    self.rust_panel.draw_ui(ctx, ui);
                }

                Panel::AMD => {
                    self.amd_panel.draw_ui(ctx, ui);
                }
            }
        });
    }
}

fn update_operator(op: &mut Operator) {
    unsafe {
        op.sign = mem::transmute(op.unsign);
        op.float = mem::transmute(op.unsign);
    }

    if is_hex(&op.hex_str) {
        op.hex_str = format!("0x{:X}", op.unsign);
    }

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
                    ui.label(format!(
                        "exp     {}",
                        op.unsign.wrapping_shl(1).wrapping_shr(24) as i32 - 127
                    ));
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
            if is_hex(&op.hex_str) {
                op.unsign = u32::from_str_radix(&op.hex_str[2..], 16).unwrap();
            }
        }
        let response = ui.add(egui::TextEdit::singleline(&mut op.unsign_str).desired_width(100.0));
        if response.changed() {
            op.unsign = op.unsign_str.parse().unwrap();
        }
        let response = ui.add(egui::TextEdit::singleline(&mut op.sign_str).desired_width(100.0));
        if response.changed() {
            let u: i32 = op.sign_str.parse().unwrap();
            op.unsign = unsafe { mem::transmute(u) };
        }
        let response = ui.add(egui::TextEdit::singleline(&mut op.float_str).desired_width(350.0));
        if response.changed() {
            let f: f32 = op.float_str.parse().unwrap();
            op.unsign = unsafe { mem::transmute(f) };
        }
    });
    ui.horizontal(|ui| {
        if ui.button("-1").clicked() {
            op.unsign = unsafe { mem::transmute(-1) };
        }
        if ui.button("0").clicked() {
            op.unsign = 0;
        }

        if ui.button("0.5").clicked() {
            op.unsign = 0.5_f32.to_bits();
        }

        if ui.button("1.0").clicked() {
            op.unsign = 1.0_f32.to_bits();
        }
        if ui.button("inf").clicked() {
            op.unsign = f32::INFINITY.to_bits();
        }

        if ui.button("nan").clicked() {
            op.unsign = f32::NAN.to_bits();
        }

        if ui.button("max").clicked() {
            op.unsign = f32::MAX.to_bits();
        }

        if ui.button("min").clicked() {
            op.unsign = f32::MIN.to_bits();
        }

        if ui.button("down").clicked() {
            op.unsign = op.float.next_down().to_bits();
        }

        if ui.button("up").clicked() {
            op.unsign = op.float.next_up().to_bits();
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
