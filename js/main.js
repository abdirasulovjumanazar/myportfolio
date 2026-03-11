/**
 * Jumanazar Portfolio - Dynamic Interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    console.log('Portfolio initialized');

    // ─── STICKY HEADER ───
    const header = document.getElementById('header');
    const handleScroll = () => {
        if (window.scrollY > 50) {
            header.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
        }
    };
    window.addEventListener('scroll', handleScroll);

    // ─── REVEAL ON SCROLL ───
    const reveal = () => {
        const reveals = document.querySelectorAll('.reveal');
        const triggerBottom = (window.innerHeight / 5) * 4;

        reveals.forEach(el => {
            const revealTop = el.getBoundingClientRect().top;
            if (revealTop < triggerBottom) {
                el.classList.add('active');
            }
        });
    };
    window.addEventListener('scroll', reveal);
    reveal(); // Initial check

    // ─── ACTIVE NAV LINK ───
    const sections = document.querySelectorAll('section');
    const navLinks = document.querySelectorAll('.nav-links a');

    window.addEventListener('scroll', () => {
        let current = '';
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.clientHeight;
            if (window.scrollY >= sectionTop - 150) {
                current = section.getAttribute('id');
            }
        });

        navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href').includes(current)) {
                link.classList.add('active');
            }
        });
    });

    // ─── MOUSE PERSPECTIVE (Subtle) ───
    const hero = document.querySelector('.section-hero');
    if (hero) {
        hero.addEventListener('mousemove', (e) => {
            const glow = document.querySelector('.hero-glow');
            const x = (e.clientX / window.innerWidth - 0.5) * 50;
            const y = (e.clientY / window.innerHeight - 0.5) * 50;
            
            glow.style.transform = `translate(${x}px, ${y}px)`;
        });
    }

    // ─── SMOOTH SCROLL FOR ALL ANCHORS ───
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            const href = this.getAttribute('href');
            if (href === '#') return;
            
            e.preventDefault();
            const target = document.querySelector(href);
            if (target) {
                const headerOffset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
});
