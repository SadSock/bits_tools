#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(float_next_up_down)]
#![feature(unchecked_math)]

use eframe::egui;
use libloading::{Library, Symbol};
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

#[derive(PartialEq, Eq)]
enum Panel {
    Rust,
    AMD,
}

#[derive(Clone, PartialEq)]
struct Operator {
    bits: [bool; 32],
    u32: u32,
    hex_str: String,
    u32_str: String,
    i32_str: String,
    f32_str: String,
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
        let instructions = ["fma_f32", "mul_u32", "mul_i32", "add_u32", "add_i32"];
        egui::ComboBox::from_label("Select Instruction!").show_index(
            ui,
            &mut self.selected,
            instructions.len(),
            |i| instructions[i],
        );

        window_operator(ctx, ui, &mut self.src0);
        window_operator(ctx, ui, &mut self.src1);

        if instructions[self.selected] == "fma_f32" {
            window_operator(ctx, ui, &mut self.src2);
        }

        match instructions[self.selected] {
            // "mul_u32" => {
            //     let a = self.src0.data;
            //     let b = self.src1.data;
            //     let c = a.wrapping_mul(b);
            //     self.dest.data = c;
            // }
            // "mul_i32" => {
            //     let a = self.src0.sign;
            //     let b = self.src1.sign;
            //     let c = a.wrapping_mul(b);
            //     unsafe {
            //         self.dest.data = mem::transmute(c);
            //     }
            // }
            // "add_u32" => {
            //     let a = self.src0.data;
            //     let b = self.src1.data;
            //     let c = a.wrapping_add(b);
            //     self.dest.data = c;
            // }
            // "add_i32" => {
            //     let a = self.src0.sign;
            //     let b = self.src1.sign;
            //     let c = a.wrapping_add(b);
            //     unsafe {
            //         self.dest.data = mem::transmute(c);
            //     }
            // }
            "fma_f32" => {
                let a: f32 = unsafe { mem::transmute(self.src0.u32) };
                let b: f32 = unsafe { mem::transmute(self.src1.u32) };
                let c: f32 = unsafe { mem::transmute(self.src2.u32) };
                let d = a.mul_add(b, c);
                self.dest.u32 = unsafe { mem::transmute(d) };
            }
            _ => {}
        }

        window_operator(ctx, ui, &mut self.dest);
    }
}

impl Default for Rust {
    fn default() -> Self {
        Self {
            src0: Operator {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src0".to_owned(),
            },
            src1: Operator {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src1".to_owned(),
            },
            src2: Operator {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src2".to_owned(),
            },
            dest: Operator {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
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
        let instructions = ["fma_f32"];
        egui::ComboBox::from_label("Select Instruction!").show_index(
            ui,
            &mut self.selected,
            instructions.len(),
            |i| instructions[i],
        );

        window_operator(ctx, ui, &mut self.src0);
        window_operator(ctx, ui, &mut self.src1);

        if instructions[self.selected] == "fma_f32" {
            window_operator(ctx, ui, &mut self.src2);
        }

        match instructions[self.selected] {
            "fma_f32" => {
                let a = self.src0.u32;
                let b = self.src1.u32;
                let c = self.src2.u32;

                unsafe {
                    let lib = libloading::Library::new("./librocm.so").unwrap();
                    let fma_f32: libloading::Symbol<unsafe extern "C" fn(u32, u32, u32) -> u32> =
                        lib.get(b"fma_f32").unwrap();
                    let d = fma_f32(a, b, c);
                    self.dest.u32 = mem::transmute(d);
                }
            }
            _ => {}
        }

        window_operator(ctx, ui, &mut self.dest);
    }
}

impl Default for AMD {
    fn default() -> Self {
        Self {
            src0: Operator {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src0".to_owned(),
            },
            src1: Operator {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src1".to_owned(),
            },
            src2: Operator {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src2".to_owned(),
            },
            dest: Operator {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
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
                                    op.u32 = op.u32 | (0x1 << (31 - i));
                                } else {
                                    op.u32 = op.u32 & !(0x1 << (31 - i));
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
                        op.u32.wrapping_shl(1).wrapping_shr(24) as i32 - 127
                    ));
                    ui.horizontal(|ui| {
                        for i in 1..9 {
                            ui.vertical(|ui| {
                                if ui.checkbox(&mut op.bits[31 - i], "").clicked() {
                                    if op.bits[31 - i] {
                                        op.u32 = op.u32 | (0x1 << (31 - i));
                                    } else {
                                        op.u32 = op.u32 & !(0x1 << (31 - i));
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
                                        op.u32 = op.u32 | (0x1 << (31 - i));
                                    } else {
                                        op.u32 = op.u32 & !(0x1 << (31 - i));
                                    }
                                }
                                ui.label(format!("{}", 31 - i));
                            });
                        }
                    });
                });
            });
        });

        ui.horizontal(|ui| {
            //hex text edit
            let res_hex = ui.add(egui::TextEdit::singleline(&mut op.hex_str).desired_width(100.0));
            if res_hex.changed() {
                if let Ok(value) = u32::from_str_radix(&op.hex_str[2..], 16) {
                    op.u32 = value;
                }
            }
            if !res_hex.has_focus() {
                op.hex_str = format!("0x{:X}", op.u32);
            }

            // u32 text edit
            let res_unsign =
                ui.add(egui::TextEdit::singleline(&mut op.u32_str).desired_width(100.0));
            if res_unsign.changed() {
                if let Ok(value) = op.u32_str.parse::<u32>() {
                    op.u32 = value;
                }
            }
            if !res_unsign.has_focus() {
                op.u32_str = format!("{}", op.u32);
            }

            //i32 text edit
            let res_sign = ui.add(egui::TextEdit::singleline(&mut op.i32_str).desired_width(100.0));
            if res_sign.changed() {
                if let Ok(value) = op.i32_str.parse::<i32>() {
                    op.u32 = unsafe { mem::transmute(value) };
                }
            }

            if !res_sign.has_focus() {
                let tmp_i32: i32 = unsafe { mem::transmute(op.u32) };
                op.i32_str = tmp_i32.to_string();
            }

            //f32 text edit
            let res_float =
                ui.add(egui::TextEdit::singleline(&mut op.f32_str).desired_width(350.0));
            if res_float.changed() {
                if let Ok(value) = op.f32_str.parse::<f32>() {
                    op.u32 = value.to_bits();
                }
            }

            if !res_float.has_focus() {
                let tmp_f32: f32 = unsafe { mem::transmute(op.u32) };
                op.f32_str = tmp_f32.to_string();
            }

            //bits
            for i in 0..32 {
                if op.u32 & (0x1 << i) != 0 {
                    op.bits[i] = true;
                } else {
                    op.bits[i] = false;
                }
            }
        });
        ui.horizontal(|ui| {
            if ui.button("-1").clicked() {
                op.u32 = unsafe { mem::transmute(-1) };
            }
            if ui.button("0").clicked() {
                op.u32 = 0;
            }

            if ui.button("0.5").clicked() {
                op.u32 = 0.5_f32.to_bits();
            }

            if ui.button("1.0").clicked() {
                op.u32 = 1.0_f32.to_bits();
            }
            if ui.button("inf").clicked() {
                op.u32 = f32::INFINITY.to_bits();
            }

            if ui.button("nan").clicked() {
                op.u32 = f32::NAN.to_bits();
            }

            if ui.button("max").clicked() {
                op.u32 = f32::MAX.to_bits();
            }

            if ui.button("min").clicked() {
                op.u32 = f32::MIN.to_bits();
            }

            if ui.button("-1ulp").clicked() {
                let tmp_f32: f32 = unsafe { mem::transmute(op.u32) };
                op.u32 = tmp_f32.next_down().to_bits();
            }

            if ui.button("+1ulp").clicked() {
                let tmp_f32: f32 = unsafe { mem::transmute(op.u32) };
                op.u32 = tmp_f32.next_up().to_bits();
            }

            if ui.button("-1").clicked() {
                if op.u32 == 0 {
                    op.u32 = u32::MAX;
                } else {
                    op.u32 = op.u32 - 1;
                }
            }
            if ui.button("+1").clicked() {
                if op.u32 == u32::MAX {
                    op.u32 = 0;
                } else {
                    op.u32 = op.u32 + 1;
                }
            }

            if ui.button("lshr").clicked() {
                op.u32 = op.u32.checked_shr(1).unwrap();
            }

            if ui.button("lshl").clicked() {
                op.u32 = op.u32.checked_shl(1).unwrap();
            }

            if ui.button("ashr").clicked() {
                unsafe {
                    let mut tmp: i32 = mem::transmute(op.u32);
                    tmp = tmp.checked_shr(1).unwrap();
                    op.u32 = mem::transmute(tmp);
                }
            }
        });
    });
}
