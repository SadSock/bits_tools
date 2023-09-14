#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")] // hide console window on Windows in release
#![feature(float_next_up_down)]
#![feature(unchecked_math)]

use eframe::egui;
use eframe::egui::{Color32};
use egui::{TextFormat, Vec2, Visuals};
use half::f16;
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
    Bits,
}

#[derive(Clone, PartialEq)]
struct OpB16 {
    bits: [bool; 16],
    u16: u16,
    hex_str: String,
    u16_str: String,
    i16_str: String,
    f16_str: String,
    name: String,
}

#[derive(Clone, PartialEq)]
struct OpB32 {
    bits: [bool; 32],
    u32: u32,
    hex_str: String,
    u32_str: String,
    i32_str: String,
    f32_str: String,
    name: String,
}

#[derive(Clone, PartialEq)]
struct OpB64 {
    bits: [bool; 64],
    u64: u64,
    hex_str: String,
    u64_str: String,
    i64_str: String,
    f64_str: String,
    name: String,
}

#[derive(Clone, PartialEq)]
struct Bits {
    src0: OpB16,
    src1: OpB32,
    src2: OpB64,
}

#[derive(Clone, PartialEq)]
struct Rust {
    src0: OpB32,
    src1: OpB32,
    src2: OpB32,
    dest: OpB32,
    selected: usize,
}

impl Default for Bits {
    fn default() -> Self {
        Self {
            src0: OpB16 {
                bits: [false; 16],
                u16: 0,
                hex_str: "0x0".to_owned(),
                u16_str: 0.to_string(),
                i16_str: 0.to_string(),
                f16_str: 0.0.to_string(),
                name: "F16".to_owned(),
            },
            src1: OpB32 {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "F32".to_owned(),
            },
            src2: OpB64 {
                bits: [false; 64],
                u64: 0,
                hex_str: "0x0".to_owned(),
                u64_str: 0.to_string(),
                i64_str: 0.to_string(),
                f64_str: 0.0.to_string(),
                name: "F64".to_owned(),
            },
        }
    }
}

impl Bits {
    fn draw_ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        draw_src_b16(ctx, ui, &mut self.src0);
        draw_src_b32(ctx, ui, &mut self.src1);
        draw_src_b64(ctx, ui, &mut self.src2);
    }
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

        ui.collapsing(&self.src0.name.to_owned(), |ui| {
            draw_src_b32(ctx, ui, &mut self.src0)
        });
        draw_src_b32(ctx, ui, &mut self.src1);

        if instructions[self.selected] == "fma_f32" {
            draw_src_b32(ctx, ui, &mut self.src2);
        }

        match instructions[self.selected] {
            "mul_u32" => {
                let a = self.src0.u32;
                let b = self.src1.u32;
                let c = a.wrapping_mul(b);
                self.dest.u32 = c;
            }
            "mul_i32" => {
                let a: i32 = unsafe { mem::transmute(self.src0.u32) };
                let b: i32 = unsafe { mem::transmute(self.src1.u32) };
                let c = a.wrapping_mul(b);
                self.dest.u32 = unsafe { mem::transmute(c) };
            }
            "add_u32" => {
                let a = self.src0.u32;
                let b = self.src1.u32;
                let c = a.wrapping_add(b);
                self.dest.u32 = c;
            }
            "add_i32" => {
                let a: i32 = unsafe { mem::transmute(self.src0.u32) };
                let b: i32 = unsafe { mem::transmute(self.src1.u32) };
                let c: i32 = a.wrapping_add(b);
                self.dest.u32 = unsafe { mem::transmute(c) };
            }
            "fma_f32" => {
                let a: f32 = unsafe { mem::transmute(self.src0.u32) };
                let b: f32 = unsafe { mem::transmute(self.src1.u32) };
                let c: f32 = unsafe { mem::transmute(self.src2.u32) };
                let d = a.mul_add(b, c);
                self.dest.u32 = unsafe { mem::transmute(d) };
            }
            _ => {}
        }

        draw_dest(ctx, ui, &mut self.dest);
    }
}

impl Default for Rust {
    fn default() -> Self {
        Self {
            src0: OpB32 {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src0".to_owned(),
            },
            src1: OpB32 {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src1".to_owned(),
            },
            src2: OpB32 {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src2".to_owned(),
            },
            dest: OpB32 {
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
    src0: OpB32,
    src1: OpB32,
    src2: OpB32,
    dest: OpB32,
    selected: usize,
}

impl AMD {
    fn draw_ui(&mut self, ctx: &egui::Context, ui: &mut egui::Ui) {
        let instructions = ["v_fma_f32", "v_mul_lo_u32", "v_add_u32", "v_add_i32"];
        egui::ComboBox::from_label("Select Instruction!").show_index(
            ui,
            &mut self.selected,
            instructions.len(),
            |i| instructions[i],
        );

        draw_src_b32(ctx, ui, &mut self.src0);
        draw_src_b32(ctx, ui, &mut self.src1);

        if instructions[self.selected] == "v_fma_f32" {
            draw_src_b32(ctx, ui, &mut self.src2);
        }

        let lib = unsafe { libloading::Library::new("./librocm.so").unwrap() };
        match instructions[self.selected] {
            "v_fma_f32" => {
                let a = self.src0.u32;
                let b = self.src1.u32;
                let c = self.src2.u32;

                unsafe {
                    let fma_f32: libloading::Symbol<unsafe extern "C" fn(u32, u32, u32) -> u32> =
                        lib.get(b"v_fma_f32").unwrap();
                    let d = fma_f32(a, b, c);
                    self.dest.u32 = d;
                }
            }
            "v_mul_lo_u32" => {
                let a = self.src0.u32;
                let b = self.src1.u32;

                unsafe {
                    let v_mul_lo_u32: libloading::Symbol<unsafe extern "C" fn(u32, u32) -> u32> =
                        lib.get(b"v_mul_lo_u32").unwrap();
                    let d = v_mul_lo_u32(a, b);
                    self.dest.u32 = d;
                }
            }
            "v_add_u32" => {
                let a = self.src0.u32;
                let b = self.src1.u32;

                unsafe {
                    let v_add_u32: libloading::Symbol<unsafe extern "C" fn(u32, u32) -> u32> =
                        lib.get(b"v_add_u32").unwrap();
                    let d = v_add_u32(a, b);
                    self.dest.u32 = d;
                }
            }
            "v_add_i32" => {
                let a = self.src0.u32;
                let b = self.src1.u32;

                unsafe {
                    let v_add_i32: libloading::Symbol<unsafe extern "C" fn(u32, u32) -> u32> =
                        lib.get(b"v_add_i32").unwrap();
                    let d = v_add_i32(a, b);
                    self.dest.u32 = d;
                }
            }
            _ => {}
        }

        draw_dest(ctx, ui, &mut self.dest);
    }
}

impl Default for AMD {
    fn default() -> Self {
        Self {
            src0: OpB32 {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src0".to_owned(),
            },
            src1: OpB32 {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src1".to_owned(),
            },
            src2: OpB32 {
                bits: [false; 32],
                u32: 0,
                hex_str: "0x0".to_owned(),
                u32_str: 0.to_string(),
                i32_str: 0.to_string(),
                f32_str: 0.0.to_string(),
                name: "src2".to_owned(),
            },
            dest: OpB32 {
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
    bits_panel: Bits,
}

impl Default for MyApp {
    fn default() -> Self {
        Self {
            panel: Panel::Bits,
            amd_panel: AMD::default(),
            rust_panel: Rust::default(),
            bits_panel: Bits::default(),
        }
    }
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            // use light theme
            ui.ctx().set_visuals(Visuals::light());

            ui.separator();
            ui.horizontal(|ui| {
                ui.selectable_value(&mut self.panel, Panel::Bits, "Bits");
                ui.selectable_value(&mut self.panel, Panel::Rust, "Rust");
                ui.selectable_value(&mut self.panel, Panel::AMD, "AMD");
            });
            ui.separator();

            match self.panel {
                Panel::Bits => {
                    self.bits_panel.draw_ui(ctx, ui);
                }

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

fn draw_src_b16(_ctx: &egui::Context, ui: &mut egui::Ui, op: &mut OpB16) {
    // ui.collapsing(op.name.to_owned(), |ui| {
    ui.horizontal(|ui| {
        // signed bit
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        let sign_color = Color32::from_rgb(187, 187, 255);
        ui.visuals_mut().widgets.active.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.noninteractive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.inactive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.hovered.bg_fill = sign_color;
        for i in 0..1 {
            ui.vertical(|ui| {
                ui.label(format!("{}", 15 - i));
                let res_checkbox = ui.add(egui::Checkbox::without_text(&mut op.bits[15 - i]));
                if res_checkbox.clicked() {
                    if op.bits[15 - i] {
                        op.u16 = op.u16 | (0x1 << (15 - i));
                    } else {
                        op.u16 = op.u16 & !(0x1 << (15 - i));
                    }
                }
            });
        }

        // exp bits
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        let sign_color = Color32::from_rgb(187, 255, 187);
        ui.visuals_mut().widgets.active.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.noninteractive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.inactive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.hovered.bg_fill = sign_color;
        for i in 1..6 {
            ui.vertical(|ui| {
                ui.label(format!("{}", 15 - i));
                let res_checkbox = ui.add(egui::Checkbox::without_text(&mut op.bits[15 - i]));
                if res_checkbox.clicked() {
                    if op.bits[15 - i] {
                        op.u16 = op.u16 | (0x1 << (15 - i));
                    } else {
                        op.u16 = op.u16 & !(0x1 << (15 - i));
                    }
                }
                if res_checkbox.hovered() {
                    let mut exp = op.u16.wrapping_shl(1).wrapping_shr(11) as i32;
                    if exp == 0 {
                        exp = -14;
                    } else {
                        exp = exp - 15;
                    }
                    res_checkbox.on_hover_text("exp: ".to_string() + &exp.to_string());
                }
            });
        }

        // exp mantissa
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        let sign_color = Color32::from_rgb(255, 187, 187);
        ui.visuals_mut().widgets.active.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.noninteractive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.inactive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.hovered.bg_fill = sign_color;
        for i in 6..16 {
            ui.vertical(|ui| {
                ui.label(format!("{}", 15 - i));
                let res_checkbox = ui.add(egui::Checkbox::without_text(&mut op.bits[15 - i]));
                if res_checkbox.clicked() {
                    if op.bits[15 - i] {
                        op.u16 = op.u16 | (0x1 << (15 - i));
                    } else {
                        op.u16 = op.u16 & !(0x1 << (15 - i));
                    }
                }
            });
        }
    });

    use egui::text::LayoutJob;
    let mut job = LayoutJob::default();

    let mut sign = "+";
    if op.bits[15] == true {
        sign = "-";
    }

    job.append(
        "sign  :  ",
        0.0,
        TextFormat {
            // background: Color32::from_rgb(187, 187, 255),
            ..Default::default()
        },
    );

    job.append(
        sign,
        0.0,
        TextFormat {
            background: Color32::from_rgb(187, 187, 255),
            ..Default::default()
        },
    );

    let mut exp = op.u16.wrapping_shl(1).wrapping_shr(11) as i32;
    if exp == 0 {
        exp = -14;
    } else {
        exp = exp - 15;
    }

    job.append(
        "    exp  :  ",
        0.0,
        TextFormat {
            // background: Color32::from_rgb(187, 255, 187),
            ..Default::default()
        },
    );
    job.append(
        &exp.to_string(),
        0.0,
        TextFormat {
            // font_id: FontId::proportional(10.0),
            background: Color32::from_rgb(187, 255, 187),
            // valign: Align::TOP,
            ..Default::default()
        },
    );

    job.append(
        "  mantissa  :  ",
        0.0,
        TextFormat {
            // background: Color32::from_rgb(187, 187, 255),
            ..Default::default()
        },
    );

    ui.label(job);
    ui.horizontal(|ui| {
        //hex text edit
        let res_hex = ui.add(egui::TextEdit::singleline(&mut op.hex_str).desired_width(80.0));
        if res_hex.changed() {
            if let Ok(value) = u16::from_str_radix(&op.hex_str[2..], 16) {
                op.u16 = value;
            }
        }
        if !res_hex.has_focus() {
            op.hex_str = format!("0x{:X}", op.u16);
        }

        // u32 text edit
        let res_unsign = ui.add(egui::TextEdit::singleline(&mut op.u16_str).desired_width(80.0));
        if res_unsign.changed() {
            if let Ok(value) = op.u16_str.parse::<u16>() {
                op.u16 = value;
            }
        }
        if !res_unsign.has_focus() {
            op.u16_str = format!("{}", op.u16);
        }

        //i32 text edit
        let res_sign = ui.add(egui::TextEdit::singleline(&mut op.i16_str).desired_width(80.0));
        if res_sign.changed() {
            if let Ok(value) = op.i16_str.parse::<i16>() {
                op.u16 = unsafe { mem::transmute(value) };
            }
        }

        if !res_sign.has_focus() {
            let tmp_i16: i16 = unsafe { mem::transmute(op.u16) };
            op.i16_str = tmp_i16.to_string();
        }

        //f32 text edit
        let res_float = ui.add(egui::TextEdit::singleline(&mut op.f16_str).desired_width(350.0));
        if res_float.changed() {
            if let Ok(value) = op.f16_str.parse::<f16>() {
                op.u16 = value.to_bits();
            }
        }

        if !res_float.has_focus() {
            let tmp_f16: f16 = unsafe { mem::transmute(op.u16) };
            op.f16_str = tmp_f16.to_string();
        }

        //bits
        for i in 0..16 {
            if op.u16 & (0x1 << i) != 0 {
                op.bits[i] = true;
            } else {
                op.bits[i] = false;
            }
        }
    });
    ui.horizontal(|ui| {

        if ui.button("inf").clicked() {
            op.u16 = f16::INFINITY.to_bits();
        }

        if ui.button("nan").clicked() {
            op.u16 = f16::NAN.to_bits();
        }

        if ui.button("max").clicked() {
            op.u16 = f16::MAX.to_bits();
        }

        if ui.button("min").clicked() {
            op.u16 = f16::MIN.to_bits();
        }


        if ui.button("des").clicked() {
            if op.u16 == 0 {
                op.u16 = u16::MAX;
            } else {
                op.u16 = op.u16 - 1;
            }
        }
        if ui.button("inc").clicked() {
            if op.u16 == u16::MAX {
                op.u16 = 0;
            } else {
                op.u16 = op.u16 + 1;
            }
        }

        if ui.button("lshr").clicked() {
            op.u16 = op.u16.checked_shr(1).unwrap();
        }

        if ui.button("lshl").clicked() {
            op.u16 = op.u16.checked_shl(1).unwrap();
        }

        if ui.button("ashr").clicked() {
            unsafe {
                let mut tmp: i16 = mem::transmute(op.u16);
                tmp = tmp.checked_shr(1).unwrap();
                op.u16 = mem::transmute(tmp);
            }
        }
    });
    // });
}

fn draw_src_b32(_ctx: &egui::Context, ui: &mut egui::Ui, op: &mut OpB32) {
    // ui.collapsing(op.name.to_owned(), |ui| {
    ui.horizontal(|ui| {
        // signed bit
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        let sign_color = Color32::from_rgb(187, 187, 255);
        ui.visuals_mut().widgets.active.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.noninteractive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.inactive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.hovered.bg_fill = sign_color;
        for i in 0..1 {
            ui.vertical(|ui| {
                ui.label(format!("{}", 31 - i));
                let res_checkbox = ui.add(egui::Checkbox::without_text(&mut op.bits[31 - i]));
                if res_checkbox.clicked() {
                    if op.bits[31 - i] {
                        op.u32 = op.u32 | (0x1 << (31 - i));
                    } else {
                        op.u32 = op.u32 & !(0x1 << (31 - i));
                    }
                }
            });
        }

        // exp bits
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        let sign_color = Color32::from_rgb(187, 255, 187);
        ui.visuals_mut().widgets.active.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.noninteractive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.inactive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.hovered.bg_fill = sign_color;
        for i in 1..9 {
            ui.vertical(|ui| {
                ui.label(format!("{}", 31 - i));
                let res_checkbox = ui.add(egui::Checkbox::without_text(&mut op.bits[31 - i]));
                if res_checkbox.clicked() {
                    if op.bits[31 - i] {
                        op.u32 = op.u32 | (0x1 << (31 - i));
                    } else {
                        op.u32 = op.u32 & !(0x1 << (31 - i));
                    }
                }
                if res_checkbox.hovered() {
                    let mut exp = op.u32.wrapping_shl(1).wrapping_shr(24) as i32;
                    if exp == 0 {
                        exp = -126;
                    } else {
                        exp = exp - 127;
                    }
                    res_checkbox.on_hover_text("exp: ".to_string() + &exp.to_string());
                }
            });
        }

        // exp mantissa
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        let sign_color = Color32::from_rgb(255, 187, 187);
        ui.visuals_mut().widgets.active.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.noninteractive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.inactive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.hovered.bg_fill = sign_color;
        for i in 9..32 {
            ui.vertical(|ui| {
                ui.label(format!("{}", 31 - i));
                let res_checkbox = ui.add(egui::Checkbox::without_text(&mut op.bits[31 - i]));
                if res_checkbox.clicked() {
                    if op.bits[31 - i] {
                        op.u32 = op.u32 | (0x1 << (31 - i));
                    } else {
                        op.u32 = op.u32 & !(0x1 << (31 - i));
                    }
                }
            });
        }
    });

    use egui::text::LayoutJob;
    let mut job = LayoutJob::default();

    let mut sign = "+";
    if op.bits[31] == true {
        sign = "-";
    }

    job.append(
        "sign  :  ",
        0.0,
        TextFormat {
            // background: Color32::from_rgb(187, 187, 255),
            ..Default::default()
        },
    );

    job.append(
        sign,
        0.0,
        TextFormat {
            background: Color32::from_rgb(187, 187, 255),
            ..Default::default()
        },
    );

    let mut exp = op.u32.wrapping_shl(1).wrapping_shr(24) as i32;
    if exp == 0 {
        exp = -126;
    } else {
        exp = exp - 127;
    }

    job.append(
        "    exp  :  ",
        0.0,
        TextFormat {
            // background: Color32::from_rgb(187, 255, 187),
            ..Default::default()
        },
    );
    job.append(
        &exp.to_string(),
        0.0,
        TextFormat {
            // font_id: FontId::proportional(10.0),
            background: Color32::from_rgb(187, 255, 187),
            // valign: Align::TOP,
            ..Default::default()
        },
    );

    job.append(
        "  mantissa  :  ",
        0.0,
        TextFormat {
            // background: Color32::from_rgb(187, 187, 255),
            ..Default::default()
        },
    );

    let op_f32: f32 = unsafe { mem::transmute(op.u32) };
    let mantissa: f32 = op_f32 / 2.0_f32.powi(exp);

    job.append(
        &mantissa.to_string(),
        0.0,
        TextFormat {
            background: Color32::from_rgb(255, 187, 187),
            ..Default::default()
        },
    );

    ui.label(job);
    egui::Grid::new("src_b32").num_columns(2).show(ui, |ui| {
        //hex text edit
        ui.label("Hexadecimal Representation");
        let res_hex = ui.add(egui::TextEdit::singleline(&mut op.hex_str).desired_width(375.0));
        if res_hex.changed() {
            if let Ok(value) = u32::from_str_radix(&op.hex_str[2..], 16) {
                op.u32 = value;
            }
        }
        if !res_hex.has_focus() {
            op.hex_str = format!("0x{:X}", op.u32);
        }
        ui.end_row();

        // u32 text edit
        ui.label("Unsigned Integer Representation");
        let res_unsign = ui.add(egui::TextEdit::singleline(&mut op.u32_str).desired_width(375.0));
        if res_unsign.changed() {
            if let Ok(value) = op.u32_str.parse::<u32>() {
                op.u32 = value;
            }
        }
        if !res_unsign.has_focus() {
            op.u32_str = format!("{}", op.u32);
        }
        ui.end_row();


        //i32 text edit
        ui.label("Signed Integer Representation");
        let res_sign = ui.add(egui::TextEdit::singleline(&mut op.i32_str).desired_width(375.0));
        if res_sign.changed() {
            if let Ok(value) = op.i32_str.parse::<i32>() {
                op.u32 = unsafe { mem::transmute(value) };
            }
        }

        if !res_sign.has_focus() {
            let tmp_i32: i32 = unsafe { mem::transmute(op.u32) };
            op.i32_str = tmp_i32.to_string();
        }
        ui.end_row();

        //f32 text edit
        ui.label("Float Representation");
        let res_float = ui.add(egui::TextEdit::singleline(&mut op.f32_str).desired_width(375.0));
        if res_float.changed() {
            if let Ok(value) = op.f32_str.parse::<f32>() {
                op.u32 = value.to_bits();
            }
        }

        if !res_float.has_focus() {
            let tmp_f32: f32 = unsafe { mem::transmute(op.u32) };
            op.f32_str = tmp_f32.to_string();
        }
        ui.end_row();
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

        if ui.button("des").clicked() {
            if op.u32 == 0 {
                op.u32 = u32::MAX;
            } else {
                op.u32 = op.u32 - 1;
            }
        }
        if ui.button("inc").clicked() {
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
    // });
}

fn draw_src_b64(_ctx: &egui::Context, ui: &mut egui::Ui, op: &mut OpB64) {
    // ui.collapsing(op.name.to_owned(), |ui| {
    ui.horizontal(|ui| {
        // signed bit
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        let sign_color = Color32::from_rgb(187, 187, 255);
        ui.visuals_mut().widgets.active.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.noninteractive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.inactive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.hovered.bg_fill = sign_color;
        for i in 0..1 {
            ui.vertical(|ui| {
                ui.label(format!("{}", 63 - i));
                let res_checkbox = ui.add(egui::Checkbox::without_text(&mut op.bits[63 - i]));
                if res_checkbox.clicked() {
                    if op.bits[63 - i] {
                        op.u64 = op.u64 | (0x1 << (63 - i));
                    } else {
                        op.u64 = op.u64 & !(0x1 << (63 - i));
                    }
                }
            });
        }

        // exp bits
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        let sign_color = Color32::from_rgb(187, 255, 187);
        ui.visuals_mut().widgets.active.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.noninteractive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.inactive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.hovered.bg_fill = sign_color;
        for i in 1..12 {
            ui.vertical(|ui| {
                ui.label(format!("{}", 63 - i));
                let res_checkbox = ui.add(egui::Checkbox::without_text(&mut op.bits[63 - i]));
                if res_checkbox.clicked() {
                    if op.bits[63 - i] {
                        op.u64 = op.u64 | (0x1 << (63 - i));
                    } else {
                        op.u64 = op.u64 & !(0x1 << (63 - i));
                    }
                }
                if res_checkbox.hovered() {
                    let mut exp = op.u64.wrapping_shl(1).wrapping_shr(53) as i32;
                    if exp == 0 {
                        exp = -1022;
                    } else {
                        exp = exp - 1023;
                    }
                    res_checkbox.on_hover_text("exp: ".to_string() + &exp.to_string());
                }
            });
        }

        // exp mantissa
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        let sign_color = Color32::from_rgb(255, 187, 187);
        ui.visuals_mut().widgets.active.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.noninteractive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.inactive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.hovered.bg_fill = sign_color;
        for i in 12..32 {
            ui.vertical(|ui| {
                ui.label(format!("{}", 63 - i));
                let res_checkbox = ui.add(egui::Checkbox::without_text(&mut op.bits[63 - i]));
                if res_checkbox.clicked() {
                    if op.bits[63 - i] {
                        op.u64 = op.u64 | (0x1 << (63 - i));
                    } else {
                        op.u64 = op.u64 & !(0x1 << (63 - i));
                    }
                }
            });
        }
    });

    ui.horizontal(|ui| {
        // exp mantissa
        ui.spacing_mut().item_spacing = Vec2::ZERO;
        let sign_color = Color32::from_rgb(255, 187, 187);
        ui.visuals_mut().widgets.active.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.noninteractive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.inactive.bg_fill = sign_color;
        ui.style_mut().visuals.widgets.hovered.bg_fill = sign_color;
        for i in 32..64 {
            ui.vertical(|ui| {
                ui.label(format!("{}", 63 - i));
                let res_checkbox = ui.add(egui::Checkbox::without_text(&mut op.bits[63 - i]));
                if res_checkbox.clicked() {
                    if op.bits[63 - i] {
                        op.u64 = op.u64 | (0x1 << (63 - i));
                    } else {
                        op.u64 = op.u64 & !(0x1 << (63 - i));
                    }
                }
            });
        }
    });
    use egui::text::LayoutJob;
    let mut job = LayoutJob::default();

    let mut sign = "+";
    if op.bits[63] == true {
        sign = "-";
    }

    job.append(
        "sign  :  ",
        0.0,
        TextFormat {
            // background: Color32::from_rgb(187, 187, 255),
            ..Default::default()
        },
    );

    job.append(
        sign,
        0.0,
        TextFormat {
            background: Color32::from_rgb(187, 187, 255),
            ..Default::default()
        },
    );

    let mut exp = op.u64.wrapping_shl(1).wrapping_shr(53) as i32;
    if exp == 0 {
        exp = -1022;
    } else {
        exp = exp - 1023;
    }

    job.append(
        "    exp  :  ",
        0.0,
        TextFormat {
            // background: Color32::from_rgb(187, 255, 187),
            ..Default::default()
        },
    );
    job.append(
        &exp.to_string(),
        0.0,
        TextFormat {
            // font_id: FontId::proportional(10.0),
            background: Color32::from_rgb(187, 255, 187),
            // valign: Align::TOP,
            ..Default::default()
        },
    );

    job.append(
        "  mantissa  :  ",
        0.0,
        TextFormat {
            // background: Color32::from_rgb(187, 187, 255),
            ..Default::default()
        },
    );

    let op_f64: f64 = unsafe { mem::transmute(op.u64) };
    let mantissa: f64 = op_f64 / 2.0_f64.powi(exp);

    job.append(
        &mantissa.to_string(),
        0.0,
        TextFormat {
            background: Color32::from_rgb(255, 187, 187),
            ..Default::default()
        },
    );

    ui.label(job);

    egui::Grid::new("src_b64").num_columns(2).show(ui, |ui| {
        //hex text edit
        ui.label("Hexadecimal Representation");
        let res_hex = ui.add(egui::TextEdit::singleline(&mut op.hex_str).
            desired_width(375.0));
        if res_hex.changed() {
            if let Ok(value) = u64::from_str_radix(&op.hex_str[2..], 16) {
                op.u64 = value;
            }
        }
        if !res_hex.has_focus() {
            op.hex_str = format!("0x{:X}", op.u64);
        }
        ui.end_row();

        // u32 text edit
        ui.label("Unsigned Integer Representation");
        let res_unsign = ui.add(egui::TextEdit::singleline(&mut op.u64_str).desired_width(375.0));
        if res_unsign.changed() {
            if let Ok(value) = op.u64_str.parse::<u64>() {
                op.u64 = value;
            }
        }
        if !res_unsign.has_focus() {
            op.u64_str = format!("{}", op.u64);
        }
        ui.end_row();
        //i32 text edit
        ui.label("Signed Integer Representation");
        let res_sign = ui.add(egui::TextEdit::singleline(&mut op.i64_str).desired_width(375.0));
        if res_sign.changed() {
            if let Ok(value) = op.i64_str.parse::<i64>() {
                op.u64 = unsafe { mem::transmute(value) };
            }
        }

        if !res_sign.has_focus() {
            let tmp_i64: i64 = unsafe { mem::transmute(op.u64) };
            op.i64_str = tmp_i64.to_string();
        }
        ui.end_row();
        //f32 text edit
        ui.label("Float Representation");
        let res_float = ui.add(egui::TextEdit::singleline(&mut op.f64_str).desired_width(375.0));
        if res_float.changed() {
            if let Ok(value) = op.f64_str.parse::<f64>() {
                op.u64 = value.to_bits();
            }
        }

        if !res_float.has_focus() {
            let tmp_f64: f64 = unsafe { mem::transmute(op.u64) };
            op.f64_str = tmp_f64.to_string();
        }
        ui.end_row();
    });

    //bits
    for i in 0..64 {
        if op.u64 & (0x1 << i) != 0 {
            op.bits[i] = true;
        } else {
            op.bits[i] = false;
        }
    }

    ui.horizontal(|ui| {
        if ui.button("inf").clicked() {
            op.u64 = f64::INFINITY.to_bits();
        }

        if ui.button("nan").clicked() {
            op.u64 = f64::NAN.to_bits();
        }

        if ui.button("max").clicked() {
            op.u64 = f64::MAX.to_bits();
        }

        if ui.button("min").clicked() {
            op.u64 = f64::MIN.to_bits();
        }

        if ui.button("-1ulp").clicked() {
            let tmp_f64: f64 = unsafe { mem::transmute(op.u64) };
            op.u64 = tmp_f64.next_down().to_bits();
        }

        if ui.button("+1ulp").clicked() {
            let tmp_f64: f64 = unsafe { mem::transmute(op.u64) };
            op.u64 = tmp_f64.next_up().to_bits();
        }

        if ui.button("des").clicked() {
            if op.u64 == 0 {
                op.u64 = u64::MAX;
            } else {
                op.u64 = op.u64 - 1;
            }
        }
        if ui.button("inc").clicked() {
            if op.u64 == u64::MAX {
                op.u64 = 0;
            } else {
                op.u64 = op.u64 + 1;
            }
        }

        if ui.button("lshr").clicked() {
            op.u64 = op.u64.checked_shr(1).unwrap();
        }

        if ui.button("lshl").clicked() {
            op.u64 = op.u64.checked_shl(1).unwrap();
        }

        if ui.button("ashr").clicked() {
            unsafe {
                let mut tmp: i64 = mem::transmute(op.u64);
                tmp = tmp.checked_shr(1).unwrap();
                op.u64 = mem::transmute(tmp);
            }
        }
    });
    // });
}

fn draw_dest(_ctx: &egui::Context, ui: &mut egui::Ui, op: &mut OpB32) {
    ui.collapsing(op.name.to_owned(), |ui| {
        ui.horizontal(|ui| {
            // signed bit
            ui.group(|ui| {
                ui.vertical(|ui| {
                    ui.label("sign");
                    ui.spacing_mut().item_spacing = Vec2::ZERO;
                    for i in 0..1 {
                        ui.vertical(|ui| {
                            ui.add_enabled(false, egui::Checkbox::new(&mut op.bits[32 - i], ""));
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
                        ui.spacing_mut().item_spacing = Vec2::ZERO;
                        for i in 1..9 {
                            ui.vertical(|ui| {
                                ui.add_enabled(
                                    false,
                                    egui::Checkbox::new(&mut op.bits[31 - i], ""),
                                );
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
                        ui.spacing_mut().item_spacing = Vec2::ZERO;
                        for i in 9..32 {
                            ui.vertical(|ui| {
                                ui.add_enabled(
                                    false,
                                    egui::Checkbox::new(&mut op.bits[31 - i], ""),
                                );
                                ui.label(format!("{}", 31 - i));
                            });
                        }
                    });
                });
            });
        });

        ui.horizontal(|ui| {
            //hex text edit
            let res_hex = ui.add_enabled(
                false,
                egui::TextEdit::singleline(&mut op.hex_str).desired_width(80.0),
            );
            if !res_hex.has_focus() {
                op.hex_str = format!("0x{:X}", op.u32);
            }

            // u32 text edit
            let res_unsign = ui.add_enabled(
                false,
                egui::TextEdit::singleline(&mut op.u32_str).desired_width(80.0),
            );
            if !res_unsign.has_focus() {
                op.u32_str = format!("{}", op.u32);
            }

            //i32 text edit
            let res_sign = ui.add_enabled(
                false,
                egui::TextEdit::singleline(&mut op.i32_str).desired_width(80.0),
            );

            if !res_sign.has_focus() {
                let tmp_i32: i32 = unsafe { mem::transmute(op.u32) };
                op.i32_str = tmp_i32.to_string();
            }

            //f32 text edit
            let res_float = ui.add_enabled(
                false,
                egui::TextEdit::singleline(&mut op.f32_str).desired_width(350.0),
            );

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
    });
}
