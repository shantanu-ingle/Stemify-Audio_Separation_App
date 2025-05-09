import React from 'react';
import './Header.css';

const Header = () => {
  return (
    <header className="header">
      <div className="logo">
        <img src="logo.png" alt="Logo" />
      </div>
      <nav className="nav">
        <button className="nav-button">HOW IT WORKS</button>
      </nav>
    </header>
  );
};

export default Header;
