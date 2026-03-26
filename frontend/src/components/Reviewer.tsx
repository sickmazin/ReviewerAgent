import React from 'react';
import './Reviewer.css';

const imgMain = "https://www.figma.com/api/mcp/asset/7813929b-2047-4e67-94b2-df986f71d8f3";
const imgIcon = "https://www.figma.com/api/mcp/asset/d7547dc7-9dcb-4e7d-8cb9-0ef87a49e2d6";
const imgEbay = "https://www.figma.com/api/mcp/asset/2a2ffe8d-41da-44b5-bf70-6814a9289432";
const imgAmazon = "https://www.figma.com/api/mcp/asset/4938aa4e-3f78-4130-8a1a-645b7fcd00e8";
const imgFork = "https://www.figma.com/api/mcp/asset/fb1e49f4-45a6-4393-98f1-f035dabd3a35";
const imgLocationPin = "https://www.figma.com/api/mcp/asset/d1b7df66-3a78-4a30-9a34-799359c28022";

const Reviewer: React.FC = () => {
  return (
    <div className="reviewer-main">
      <img src={imgMain} alt="" className="reviewer-bg-img" />
      
      <div className="reviewer-container">
        {/* Sidebar Left */}
        <div className="reviewer-sidebar-left">
          <h2>Old reviews</h2>
          
          <div className="reviewer-new-chat">
            <div className="reviewer-new-chat-icon">
              <img src={imgIcon} alt="" />
            </div>
            <div className="reviewer-new-chat-text">New reviews</div>
          </div>
          
          <div className="reviewer-chat-list">
            {[...Array(12)].map((_, i) => (
              <div key={i} className="reviewer-chat-item">
                <div className="reviewer-chat-item-text">Chat title....</div>
              </div>
            ))}
          </div>
        </div>

        {/* Center Main Area */}
        <div className="reviewer-center">
          <div className="reviewer-center-title">Put review here</div>
          
          <div className="reviewer-text-input-container">
            <textarea 
              className="reviewer-textarea" 
              placeholder="Review..." 
              defaultValue="Review..."
            />
          </div>
          
          <button className="reviewer-evaluate-btn">
            <span>Evaluate score</span>
          </button>
          
          <div className="reviewer-ai-output">
            <div className="reviewer-score-box">
              <div className="reviewer-score-title">Insightfulness Score</div>
              <div className="reviewer-score-value">65/100 😉</div>
            </div>
            
            <div className="reviewer-checkbox-group">
              <div className="reviewer-checkbox-item">
                <input type="checkbox" className="reviewer-checkbox" />
                <div className="reviewer-checkbox-text">Follow guide linees</div>
              </div>
              <div className="reviewer-checkbox-item">
                <input type="checkbox" className="reviewer-checkbox" />
                <div className="reviewer-checkbox-text">Grammatical errors</div>
              </div>
            </div>
          </div>
        </div>

        {/* Sidebar Right */}
        <div className="reviewer-sidebar-right">
          <div className="reviewer-sidebar-right-title">Where you post this review?</div>
          
          <div className="reviewer-website-item">
            <input type="checkbox" className="reviewer-checkbox" />
            <div className="reviewer-website-icon">
              <img src={imgAmazon} alt="Amazon" />
            </div>
            <div className="reviewer-website-text">Amazon</div>
          </div>
          
          <div className="reviewer-website-item">
            <input type="checkbox" className="reviewer-checkbox" />
            <div className="reviewer-website-icon">
              <img src={imgEbay} alt="Ebay" />
            </div>
            <div className="reviewer-website-text">Ebay</div>
          </div>
          
          <div className="reviewer-website-item">
            <input type="checkbox" className="reviewer-checkbox" />
            <div className="reviewer-website-icon">
              <img src={imgFork} alt="Restaurants" />
            </div>
            <div className="reviewer-website-text">Restaurants</div>
          </div>
          
          <div className="reviewer-website-item">
            <input type="checkbox" className="reviewer-checkbox" />
            <div className="reviewer-website-icon">
              <img src={imgLocationPin} alt="Location" />
            </div>
            <div className="reviewer-website-text">Location</div>
          </div>
          
          <div className="reviewer-space" />
          
          <div className="reviewer-sidebar-right-title">Link of product in review:</div>
          
          <div className="reviewer-url-container">
            <input 
              type="text" 
              className="reviewer-url-input" 
              placeholder="Url (optional)..." 
            />
          </div>
          
          <div className="reviewer-optional-text">(Optional)</div>
        </div>
      </div>
    </div>
  );
};

export default Reviewer;
